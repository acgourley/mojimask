"""Dump what today's renderer (gpu_draw.Compositor, through track.draw_frame) is given and what it draws, for
the native app's Editor-only mode (`Mojimask --data DIR`) and the parity tools that read out/poc/data/<clip>. It began
as the reference for the front-end proofs of concept (native/docs/stack-decision.md).

    MOJIMASK_DEPTH=anything-small PYTHONPATH=. <main>/.venv/bin/python native/tools/dump_reference.py \
        out/poc/projects/jazz out/poc/data/jazz [--ref all|0,40,80]

Writes into the out dir (git-ignored):
  meta.json         W, H, fps, n frames, video path, kernel constants, pictures (hash -> file, emoji key, size)
  kernel.metal      the compositor's Metal source, constants substituted (exactly what Python compiles)
  covers.bin        float32 rows of STRIDE floats, all frames back to back; index in frames.json
  frames.json       per frame: {t, n, offset (row index), seed, soft, mw, mh, near (file or null),
                                covers: [semantic dicts: centre, half, radius, box, slice, light, vel, occ, saturation],
                                circles: [[cx, cy, d] ...] (the frame's covers as circles, for blur/pixelate/solid)}
  atlas/<hash>.png  each distinct cover picture (RGBA, straight alpha, square, <= 384 px). Pictures change from frame
                    to frame (the emoji is tilted/turned with the head), so each frame lists its own: frames.json
                    "pics" = [hash, ...]; a cover row's P[3] (its atlas slice) is an index into that list.
  near/NNNNN.f32    depth nearness map (float32, mh x mw), when a cover on that frame is cut by depth
  in/NNNNN.png      the frame as Python's decoder gives it (video_io, color="display"), for the ref frames
  ref/NNNNN.png     today's output for that frame (emoji style, GPU path)
  ref_blur/, ref_pixelate/, ref_solid/   the CPU styles, for the ref frames (first 3 ref frames only)

Atlas slices: the kernel samples a 2D texture array (RGBA16F, 384x384, 48 slices, mipmapped). To draw frame t, put
frame t's pics[k] in slice k (pictures over 384 px are first resized to 384 with INTER_AREA), regenerate mipmaps. Each
slice holds the sprite in the top-left n x n corner, BGR order, premultiplied: rgba = (B*a, G*a, R*a, a) / 255. P[31] = n/384.
"""
import argparse
import hashlib
import json
import os
import struct
import sys

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("out")
    ap.add_argument("--ref", default="all", help="frames to save in/ref pictures for: all, or a comma list")
    a = ap.parse_args()
    proj, out = os.path.abspath(a.project), os.path.abspath(a.out)
    meta = json.load(open(f"{proj}/meta.json"))
    for d in ("atlas", "near", "in", "ref", "ref_blur", "ref_pixelate", "ref_solid"):
        os.makedirs(f"{out}/{d}", exist_ok=True)

    sys.argv = ["review_server.py", meta["video"], f"{proj}/dets", "--workdir", proj, "--original", meta["video"],
                "--port", "0"]
    from mojimask import review_server as RS  # loads the project (module-level startup), serves nothing
    from mojimask import gpu_draw as G
    from mojimask import track as T
    from mojimask import video_io
    from mojimask import depth as DP

    comp = G.get()
    assert comp is not None, "no Metal compositor"
    cap = {}
    orig_draw = G.Compositor.draw

    def draw(self, f, covers, near=None, seed=0, soft=0.15):
        res = orig_draw(self, f, covers, near, seed, soft)
        rows, sem, pics = [], [], []
        for cv in covers:
            sl, n, _ = self.slots[(id(cv["pic"]), cv["pic"].shape)]
            h = hashlib.md5(cv["pic"].tobytes()).hexdigest()[:16]
            if h not in [x[0] for x in pics]:
                pics.append((h, cv["pic"]))
            local = [x[0] for x in pics].index(h)
            r = self._cover(cv, sl, n)
            r[3] = local  # slice = index into this frame's own picture list (see the docstring)
            rows.append(r)
            sem.append({"centre": list(map(float, cv["centre"])), "half": float(cv["half"]), "radius": float(cv["radius"]),
                        "box": list(map(float, cv["box"])), "slice": local, "pic": h, "pic_size": n,
                        "light": {k: (list(map(float, v)) if hasattr(v, "__len__") else float(v)) for k, v in cv["light"].items()}
                        if cv.get("light") else None,
                        "vel": list(map(float, cv["vel"])) if cv.get("vel") else None,
                        "occ": {k: float(v) for k, v in cv["occ"].items()} if cv.get("occ") else None,
                        "saturation": float(cv.get("saturation", 1.0))})
        cap.update(rows=rows, sem=sem, pics=pics, near=near, seed=seed, soft=soft)
        return res
    G.Compositor.draw = draw

    edits = RS.load_edits()
    pf, kw = RS.render_setup(edits, {"style": "emoji"}, wait_light=True)
    N, W, H = RS.N, RS.W, RS.H
    refs = set(range(N)) if a.ref == "all" else {int(x) for x in a.ref.split(",")}
    sprites, frames, allrows = {}, [], []
    slices = {}
    for t, f in enumerate(video_io.frames(RS.A.video, RS.VMETA, color="display")):
        if t >= N:
            break
        cap.clear()
        o = T.draw_frame(f, t, pf[t], sprites, **kw)
        rows = cap.get("rows", [])
        near = cap.get("near")
        nf = None
        if near is not None:
            nf = f"near/{t:05d}.f32"
            np.ascontiguousarray(near, np.float32).tofile(f"{out}/{nf}")
        for h, pic in cap.get("pics", []):
            if h not in slices:
                key = next((k for k, v in sprites.items() if v is pic), None)
                cv2.imwrite(f"{out}/atlas/{h}.png", cv2.cvtColor(pic, cv2.COLOR_RGBA2BGRA))
                slices[h] = {"file": f"atlas/{h}.png", "key": key, "size": int(pic.shape[0])}
        # the frame's covers as circles (the CPU styles), in drawing order, minus whitelisted pieces
        items = [x for x in T.draw_order(pf[t], t, kw["occlusion"]) if not (x[3:] and x[3] in kw["shown"])]
        circles = [[float(x[1][0]), float(x[1][1]), float(x[1][2] * kw["scale"])] for x in items]
        mh, mw = near.shape if near is not None else (1, 1)
        frames.append({"t": t, "n": len(rows), "offset": len(allrows), "seed": float(cap.get("seed", t)),
                       "soft": float(cap.get("soft", DP.OCC_SOFT)), "pics": [h for h, _ in cap.get("pics", [])], "mw": mw, "mh": mh, "near": nf,
                       "covers": cap.get("sem", []), "circles": circles})
        allrows += rows
        if t in refs:
            cv2.imwrite(f"{out}/in/{t:05d}.png", f)
            cv2.imwrite(f"{out}/ref/{t:05d}.png", o)
            if t in sorted(refs)[:3]:
                for st in ("blur", "pixelate", "solid"):
                    g = f.copy()
                    for cx, cy, d in circles:
                        T.cover(g, (cx, cy, d), 1.0, st)
                    cv2.imwrite(f"{out}/ref_{st}/{t:05d}.png", g)
        if t % 20 == 0:
            print(f"t={t} covers={len(rows)} near={'y' if near is not None else 'n'}", flush=True)
    C = np.stack(allrows).astype(np.float32) if allrows else np.zeros((0, G.STRIDE), np.float32)
    C.tofile(f"{out}/covers.bin")
    json.dump(frames, open(f"{out}/frames.json", "w"))
    src = G.SRC.replace("@STRIDE@", str(G.STRIDE)).replace("@FACE_DY@", repr(float(T.FACE_DY)))
    open(f"{out}/kernel.metal", "w").write(src)
    json.dump({"W": W, "H": H, "fps": RS.FPS, "n": N, "video": os.path.abspath(RS.A.video),
               "has_audio": bool(RS.VMETA.get("has_audio")),
               "STRIDE": G.STRIDE, "TEX": G.TEX, "SLICES": G.SLICES, "SIGMA_S": G.SIGMA_S, "SIGMA_R": G.SIGMA_R,
               "FACE_ZONE": list(map(float, DP.FACE_ZONE)), "FACE_DY": float(T.FACE_DY),
               "frame_struct": "struct Frame { uint W, H, n, mw, mh; float kx, ky, seed, soft, ss, sr, z0, z1; } "
                               "kx = mw/W, ky = mh/H, ss = SIGMA_S, sr = SIGMA_R, (z0, z1) = FACE_ZONE",
               "bindings": "buffer0 image (uchar BGR, W*H*3, in place), buffer1 covers, bytes2 Frame, buffer3 near "
                           "(float mw*mh), buffer4 guide scratch (mw*mh*16 bytes), texture0 atlas; dispatch 'guide' "
                           "over mw x mh first when near is present, then 'compose' over W x H",
               "pictures": slices, "ref_frames": sorted(refs), "style_ref_frames": sorted(refs)[:3],
               "covers_total": int(len(C))}, open(f"{out}/meta.json", "w"), indent=1)
    print(f"done: {N} frames, {len(C)} cover rows, {len(slices)} pictures -> {out}", flush=True)
    os._exit(0)  # the review server's background threads (lighting) are daemons; don't wait


if __name__ == "__main__":
    main()
