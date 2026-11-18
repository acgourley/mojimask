"""Compare rendered frames with a reference (native/tools/dump_reference.py's ref/, or any folder of NNNNN.png).

    python3 native/tools/compare_frames.py out/poc/data/jazz/ref <frames dir> [--json]

Both dirs hold NNNNN.png. For each frame in both: PSNR (dB), mean abs error and max abs error (0..255), and the
share of pixels off by more than 8 levels. Needs numpy + Pillow (the project venv, or any Python that has them).
Rule of thumb: PSNR >= 45 dB and <0.1% pixels over 8 is "the same picture" (float rounding, grain); 35-45 is a
visible-on-zoom difference worth looking at; below 35 something is wrong.
"""
import json
import os
import sys

import numpy as np
from PIL import Image


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), np.float32)


def main():
    ref, got = sys.argv[1], sys.argv[2]
    as_json = "--json" in sys.argv
    rows = []
    for name in sorted(os.listdir(ref)):
        g = os.path.join(got, name)
        if not name.endswith(".png") or not os.path.exists(g):
            continue
        a, b = load(os.path.join(ref, name)), load(g)
        if a.shape != b.shape:
            rows.append({"frame": name, "error": f"size {b.shape[:2]} vs ref {a.shape[:2]}"})
            continue
        d = np.abs(a - b)
        mse = float((d ** 2).mean())
        rows.append({"frame": name, "psnr": round(99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse), 2),
                     "mae": round(float(d.mean()), 3), "max": int(d.max()), "over8_pct": round(float((d.max(-1) > 8).mean() * 100), 3)})
    ok = [r for r in rows if "psnr" in r]
    summary = {"frames": len(rows), "min_psnr": min((r["psnr"] for r in ok), default=None),
               "mean_psnr": round(float(np.mean([r["psnr"] for r in ok])), 2) if ok else None,
               "max_over8_pct": max((r["over8_pct"] for r in ok), default=None),
               "errors": [r for r in rows if "error" in r]}
    if as_json:
        print(json.dumps({"summary": summary, "frames": rows}))
    else:
        for r in rows:
            print(r)
        print("summary:", summary)


if __name__ == "__main__":
    main()
