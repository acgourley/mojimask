"""Parity: frames drawn by mm-render (mm-render-ref, from the draw records) vs Python's track.draw_frame (and
track.watermark), per config of the draw reference (draw_ref.py): every style (emoji, portrait, blur, pixelate, solid),
lighting, head pose, depth and the corner mark, on jazz frames. Target: >= 45 dB per frame (whole frame PSNR).

    native/tools/parity/draw_frames.py [--no-build] [--configs a,b] [--py-sprites] [--regions]

--py-sprites draws with Python's emoji pictures (isolates the compositor from CoreText). --regions also reports the
PSNR over just the covers' boxes (a stricter view: the whole frame dilutes errors with untouched pixels).
Rows: for emoji/portrait configs it also checks the compositor rows Rust builds from the records against Python's
(gpu_draw.Compositor._cover), field by field (posed covers differ by design in the box size fields; see cover.rs).
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import draw_common as D  # noqa: E402

TARGET = 45.0


def arg(k):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None


def main():
    if "--no-build" not in sys.argv:
        D.build()
    cfg = json.load(open(f"{D.REF}/configs.json"))
    configs = arg("--configs").split(",") if arg("--configs") else list(cfg["configs"])
    extra = ["--rows"] + (["--py-sprites"] if "--py-sprites" in sys.argv else [])
    D.run("--configs", ",".join(configs), *extra)
    out, ok = {}, True
    for c in configs:
        frames, rows_bad = {}, []
        for t in cfg["configs"][c]["frames"]:
            py = cv2.imread(f"{D.REF}/{c}/{t:05d}.png")
            rs = cv2.imread(f"{D.RUST}/{c}/{t:05d}.png")
            d = np.abs(py.astype(int) - rs.astype(int))
            r = {"psnr": round(D.psnr(py, rs), 2), "max_err": int(d.max()), "over8": round(float((d.max(-1) > 8).mean() * 100), 3)}
            rec = json.load(open(f"{D.REF}/{c}/{t:05d}.json"))
            if "--regions" in sys.argv and rec["covers"]:
                m = np.zeros(py.shape[:2], bool)
                for cv in rec["covers"]:
                    x, y, s = cv["c"]
                    h = s * rec["scale"] * 0.9
                    m[max(int(y - h), 0):int(y + h), max(int(x - h), 0):int(x + h)] = True
                r["covers_psnr"] = round(D.psnr(py[m], rs[m]), 2)
            rp = f"{D.RUST}/{c}/{t:05d}.rows.json"
            if os.path.exists(rp):
                rr = np.array(json.load(open(rp))["rows"], np.float32).reshape(-1, 36)
                for i, cv in enumerate(rec["covers"]):
                    pr = np.array(cv.get("py_row"), np.float32)
                    posed = "pose" in cv
                    skip = {3} | ({0, 1, 2, 7, 30, 31, 34, 35} if posed else set())
                    bad = [k for k in range(36) if k not in skip and not np.isclose(pr[k], rr[i][k], rtol=1e-5, atol=1e-5)]
                    if posed:  # the canvas box is the same as Python's
                        bad += [k for k in (0, 1, 2) if not np.isclose(pr[k], rr[i][k], atol=1e-4)]
                    if bad:
                        rows_bad.append({"t": t, "cover": i, "fields": {k: [float(pr[k]), float(rr[i][k])] for k in bad}})
            frames[t] = r
        ps = [f["psnr"] for f in frames.values()]
        good = min(ps) >= TARGET and not rows_bad
        ok &= good
        out[c] = {"pass": good, "min_db": min(ps), "mean_db": round(float(np.mean(ps)), 2),
                  "rows_mismatched": rows_bad[:5], "n_rows_mismatched": len(rows_bad), "frames": frames}
    D.report("draw_frames", {"target_db": TARGET, "summary": {c: [v["min_db"], v["pass"]] for c, v in out.items()},
                             "configs": out}, ok)


if __name__ == "__main__":
    main()
