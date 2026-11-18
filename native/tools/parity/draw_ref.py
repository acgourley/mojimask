"""Reference pictures and draw records for mm-render's parity tests (the draw stream, P2).

Runs the Python engine's own drawing (track.draw_frame, through gpu_draw on Metal for emoji/portrait and the CPU for
blur/pixelate/solid; track.watermark for the corner mark) on jazz frames under several configs, and writes, for each
config and frame, the picture Python drew and the RECORD the Rust renderer draws from (what the engine knows about each
cover, before any drawing decision): the head box, the emoji key, the head pose, the lighting estimate, the depth cut.

    MOJIMASK_DEPTH=anything-small PYTHONPATH=<ROOT> <main>/.venv/bin/python native/tools/parity/draw_ref.py \
        out/native/draw/project out/native/draw/ref [--frames 0,16,...]

Out dir (under out/, git-ignored):
  in/NNNNN.png              the frame as video_io gives it (color="display")
  <config>/NNNNN.png        Python's output; <config>/NNNNN.json the record (see records below)
  near/NNNNN.f32            the depth guide (float32 mh x mw) when a cover is cut on that frame (depth configs)
  sprites/<key>.png         emoji_sprite(key) (RGBA straight), every key used plus track.EMOJI; key = code points hex
  configs.json              the configs, and the constants (TEX, SLICES, SIGMA_S/R, FACE_ZONE, FACE_DY, CANVAS ...)
"""
import argparse
import hashlib
import json
import os
import sys

import cv2
import numpy as np
import subprocess

# the main checkout (a worktree has no venv or models of its own): the parent of git's common directory
MAIN_CHECKOUT = os.path.dirname(subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse",
                                                "--path-format=absolute", "--git-common-dir"],
                                               capture_output=True, text=True).stdout.strip())

MAIN = MAIN_CHECKOUT

CONFIGS = {
    # name: (style, light, pose, depth, light setting overrides, mark)
    "emoji_flat": ("emoji", False, False, False, {}, False),            # the project's saturation (0 on jazz)
    "emoji_flat_sat1": ("emoji", False, False, False, {"saturation": 1.0}, False),
    "emoji_lit": ("emoji", True, False, False, {}, False),
    "emoji_lit_colour": ("emoji", True, False, False, {"saturation": 1.0, "light_colour": [30, 0.6], "level": 1.2,
                                                         "black": 0.5, "shutter": 0.5}, False),
    "emoji_pose": ("emoji", False, True, False, {"saturation": 1.0}, False),
    "emoji_depth": ("emoji", False, False, True, {"saturation": 1.0}, False),
    "emoji_full": ("emoji", True, True, True, {}, False),                # the app's export on jazz with depth on
    "portrait": ("portrait", True, True, False, {}, False),
    "portrait_flat": ("portrait", False, True, False, {"saturation": 1.0}, False),
    "blur": ("blur", False, False, False, {}, False),
    "pixelate": ("pixelate", False, False, False, {}, False),
    "solid": ("solid", False, False, False, {}, False),
    "blur_depth": ("blur", False, False, True, {}, False),
    "mark": ("emoji", False, False, False, {"saturation": 1.0}, True),
}


def key_name(ch):
    return "-".join(f"{ord(c):x}" for c in ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("out")
    ap.add_argument("--frames", default="0,16,32,48,64,80,96,112,126")
    ap.add_argument("--configs", default="all")
    a = ap.parse_args()
    proj, out = os.path.abspath(a.project), os.path.abspath(a.out)
    meta = json.load(open(f"{proj}/meta.json"))
    want = sorted(int(x) for x in a.frames.split(","))
    configs = CONFIGS if a.configs == "all" else {k: CONFIGS[k] for k in a.configs.split(",")}
    for d in ["in", "near", "sprites", *configs]:
        os.makedirs(f"{out}/{d}", exist_ok=True)

    sys.argv = ["review_server.py", meta["video"], f"{proj}/dets", "--workdir", proj, "--original", meta["video"],
                "--port", "0"]
    from mojimask import portrait as PT
    _own = PT.own_pose

    def own_pose(ch):  # the main checkout's cached pose of the same (md5-identical) default photo: no RetinaFace here
        src = PT._path(ch)
        cached = os.path.join(MAIN, "web", "assets", os.path.basename(src or "") + ".pose.json")
        if src and os.path.exists(cached) and \
                hashlib.md5(open(src, "rb").read()).digest() == \
                hashlib.md5(open(os.path.join(MAIN, "web", "assets", os.path.basename(src)), "rb").read()).digest():
            d = json.load(open(cached))
            return d["roll"], d["yaw"]
        return _own(ch)
    PT.own_pose = own_pose

    from mojimask import review_server as RS
    from mojimask import gpu_draw as G
    from mojimask import track as T
    from mojimask import video_io
    from mojimask import depth as DP
    from mojimask import lighting as LT
    from mojimask import head_pose as HP

    assert G.get() is not None, "no Metal compositor"
    cap = {}
    orig_draw = G.Compositor.draw

    def draw(self, f, covers, near=None, seed=0, soft=0.15):
        res = orig_draw(self, f, covers, near, seed, soft)
        rows = []
        for cv in covers:
            sl, n, _ = self.slots[(id(cv["pic"]), cv["pic"].shape)]
            r = self._cover(cv, sl, n)
            r[3] = -1
            rows.append([float(x) for x in r])
        cap.update(rows=rows, near=near, covers=covers)
        return res
    G.Compositor.draw = draw

    base_edits = RS.load_edits()
    frames = {}
    for t, f in enumerate(video_io.frames(RS.A.video, RS.VMETA, color="display")):
        if t >= RS.N:
            break
        if t in want:
            frames[t] = f
            cv2.imwrite(f"{out}/in/{t:05d}.png", f)
    keys_used = set(T.EMOJI)
    summary = {}
    for name, (style, light_on, pose_on, depth_on, over, mark) in configs.items():
        edits = json.loads(json.dumps(base_edits))
        edits["light"] = {**(edits.get("light") or {}), **over, "enabled": light_on}
        edits["head_pose"] = pose_on
        pf, kw = RS.render_setup(edits, {"style": style}, wait_light=True)
        if not depth_on:
            kw["occlusion"] = None
        else:
            assert kw["occlusion"] is not None, "depth config needs MOJIMASK_DEPTH=anything-small and a depth map"
        st = LT.settings(edits)
        light = kw["light"]
        nf = 0
        for t, f in frames.items():
            cap.clear()
            o = T.draw_frame(f, t, pf[t], {} if False else SPRITES, **kw)
            if mark:
                o = T.watermark(o)
            cv2.imwrite(f"{out}/{name}/{t:05d}.png", o)
            occl = kw["occlusion"]
            items = [x for x in T.draw_order(pf[t], t, occl) if not (x[3:] and x[3] in kw["shown"])]
            near, occ = occl.plan(t, items, kw["scale"], f.shape) if occl is not None else (None, {})
            near_file = None
            if near is not None:
                near_file = f"near/{t:05d}.f32"
                np.ascontiguousarray(near, np.float32).tofile(f"{out}/{near_file}")
            covers = []
            for i, (tid, c, kind, *rest) in enumerate(items):
                piece = rest[0] if rest else tid
                rec = {"tid": str(tid), "piece": str(piece), "c": [float(v) for v in c[:3]]}
                if style in ("emoji", "portrait"):
                    if style == "portrait":
                        ch = kw["portrait"] or PT.PORTRAIT_KEY
                    else:
                        ch = T.emoji_for(tid, kw["overrides"])
                    keys_used.add(ch)
                    rec["key"] = ch
                    rec["key_hex"] = key_name(ch)
                    hp = (kw["head_pose"] or {}).get(piece, {}).get(t)
                    if hp is not None:
                        rec["pose"] = [float(hp[0]), float(hp[1]), int(hp[2])]
                        r_, y_, m_ = HP.sprite_pose(ch, *hp)
                        rec["sprite_pose"] = [float(r_), float(y_), bool(m_)]
                    if style == "portrait":
                        rec["own_pose"] = list(map(float, PT.own_pose(ch)))
                    if light is not None:
                        k, g, amb = light.L.auto(piece, t)
                        rec["light_auto"] = {"key": list(map(float, k)), "grad": list(map(float, g)),
                                             "amb": list(map(float, amb))}
                        rec["vel_raw"] = list(map(float, light.velocity(piece, t)))
                        lp = light.params(piece, t, f)
                        rec["py_params"] = {"key": list(map(float, lp["key"])), "amb": list(map(float, lp["amb"])),
                                            "grad": list(map(float, lp["grad"])), "black": lp["black"],
                                            "saturation": lp["saturation"], "grain": lp["grain"]}
                    oc = occ.get(i)
                    if oc is not None:
                        rec["occ"] = {k2: float(v) for k2, v in oc.items()}
                    if cap.get("rows"):
                        rec["py_row"] = cap["rows"][i]
                else:
                    oc = occ.get(i)
                    if oc is not None:
                        rec["occ"] = {k2: float(v) for k2, v in oc.items()}
                covers.append(rec)
            if style in ("emoji", "portrait"):
                assert len(cap.get("rows", [])) == len(covers), (name, t, len(cap.get("rows", [])), len(covers))
            fr = {"t": t, "style": style, "scale": float(kw["scale"]), "seed": float(t), "soft": float(DP.OCC_SOFT),
                  "saturation": float(kw["saturation"]), "light": None, "near": near_file,
                  "mw": int(near.shape[1]) if near is not None else 1, "mh": int(near.shape[0]) if near is not None else 1,
                  "mark": mark, "covers": covers}
            if light is not None:
                fr["light"] = {"settings": {k2: v for k2, v in light.st.items()}, "film": list(light.L.film(t)),
                               "grain": float(LT.frame_grain(f)) if light.st["grain"] else 0.0}
            json.dump(fr, open(f"{out}/{name}/{t:05d}.json", "w"))
            nf += 1
        summary[name] = {"style": style, "light": light_on, "pose": pose_on, "depth": depth_on, "settings": over,
                         "mark": mark, "frames": sorted(frames)}
        print(f"{name}: {nf} frames", flush=True)
    for ch in sorted(keys_used):
        sp = T.emoji_sprite(ch)
        cv2.imwrite(f"{out}/sprites/{key_name(ch)}.png", cv2.cvtColor(sp, cv2.COLOR_RGBA2BGRA))
    from mojimask import about
    json.dump({"configs": summary, "W": RS.W, "H": RS.H, "TEX": G.TEX, "SLICES": G.SLICES, "SIGMA_S": G.SIGMA_S,
               "SIGMA_R": G.SIGMA_R, "FACE_ZONE": list(map(float, DP.FACE_ZONE)), "FACE_DY": float(T.FACE_DY),
               "OCC_SOFT": DP.OCC_SOFT, "CANVAS": HP.CANVAS, "keys": {key_name(k): k for k in sorted(keys_used)},
               "portrait_photo": PT._path(PT.PORTRAIT_KEY), "eyes": os.path.join(os.path.dirname(T.__file__), "..", "web",
                                                                               "emoji_eyes.json"),
               "brand": about.BRAND, "font_file": about.FONT_FILE, "name": about.NAME,
               "MARK_SIZE": T.MARK_SIZE, "MARK_ALPHA": T.MARK_ALPHA},
              open(f"{out}/configs.json", "w"), indent=1, ensure_ascii=False)
    print("done", flush=True)
    os._exit(0)


SPRITES = {}

if __name__ == "__main__":
    main()
