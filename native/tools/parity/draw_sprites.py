"""Parity: emoji pictures drawn in Rust (CoreText through AppKit, mm_render::sprites::emoji) vs Python's
track.emoji_sprite, for every emoji the draw reference has (track.EMOJI plus those jazz uses). Target >= 50 dB.

    native/tools/parity/draw_sprites.py [--no-build]

Needs the reference (draw_ref.py). PSNR over RGBA with the colour premultiplied (colour under alpha 0 is meaningless).
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import draw_common as D  # noqa: E402

TARGET = 50.0


def premul(p):
    a = p[..., 3:4].astype(np.float64)
    return np.concatenate([p[..., :3] * a / 255.0, a], -1)


def main():
    if "--no-build" not in sys.argv:
        D.build()
    D.run("--sprites")
    res, worst = {}, 99.0
    for f in sorted(os.listdir(f"{D.REF}/sprites")):
        py = cv2.imread(f"{D.REF}/sprites/{f}", cv2.IMREAD_UNCHANGED)
        rp = f"{D.RUST}/sprites/{f}"
        if not os.path.exists(rp):
            continue  # portraits: compared through the frames
        rs = cv2.imread(rp, cv2.IMREAD_UNCHANGED)
        if rs.shape != py.shape:
            res[f] = {"psnr": 0.0, "shape": [list(py.shape), list(rs.shape)]}
            worst = 0.0
            continue
        p = D.psnr(premul(py), premul(rs))
        res[f[:-4]] = {"psnr": round(p, 2), "alpha_identical": bool((py[..., 3] == rs[..., 3]).all()),
                       "max_err": int(np.abs(premul(py) - premul(rs)).max())}
        worst = min(worst, p)
    vals = [v["psnr"] for v in res.values()]
    D.report("draw_sprites", {"target_db": TARGET, "n": len(vals), "min_db": round(worst, 2),
                              "mean_db": round(float(np.mean(vals)), 2), "sprites": res}, worst >= TARGET and vals)


if __name__ == "__main__":
    main()
