"""mm-vision against cv2: every case vision_dump.py writes must come out identical (0 differing pixels).

The one stated exception: cv2.filter2D with a kernel of 50 taps or more, which cv2 computes by DFT on arm64;
there the tolerance is 1e-6 absolute (float rounding only). Prints one JSON object; the last line is
{"pass": true|false}.

    PYTHONPATH=ROOT .venv/bin/python native/tools/parity/vision_check.py            # dump, build, check
    PYTHONPATH=ROOT .venv/bin/python native/tools/parity/vision_check.py --bench    # also time both on 4K frames
    options: --no-dump (reuse the dump), --filter <substring> (only matching cases)
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
NATIVE = os.path.join(ROOT, "native")
OUT = os.path.join(ROOT, "out", "native", "vision_parity")
DFT_TOL = 1e-6


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def bench_cv2():
    import cv2
    import numpy as np

    bgr = np.load(os.path.join(OUT, "bench_bgr4k.npy"))
    gray = np.load(os.path.join(OUT, "bench_gray4k.npy"))
    gf = np.load(os.path.join(OUT, "bench_grayf4k.npy"))
    bf = np.load(os.path.join(OUT, "bench_bgrf4k.npy"))
    m = np.array([[0.9, -0.2, 100.0], [0.2, 0.9, -50.0]])
    y2b = np.array([[1.1642, 2.0167, 0.0, -278.2338], [1.1645, -0.3897, -0.8121, 133.9585], [1.1642, 0.0, 1.5956, -224.693]], np.float32)
    k9 = np.full((9, 9), 1 / 81, np.float32)
    ops = {
        "cvt_bgr2gray": lambda: cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
        "cvt_bgr2rgb": lambda: cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        "resize_linear_to1280": lambda: cv2.resize(bgr, (1280, 720), interpolation=cv2.INTER_LINEAR),
        "resize_area_to1280": lambda: cv2.resize(bgr, (1280, 720), interpolation=cv2.INTER_AREA),
        "resize_area_to1000": lambda: cv2.resize(bgr, (1000, 562), interpolation=cv2.INTER_AREA),
        "resize_cubic_to1280": lambda: cv2.resize(bgr, (1280, 720), interpolation=cv2.INTER_CUBIC),
        "resize_nearest_half": lambda: cv2.resize(bgr, (1920, 1080), interpolation=cv2.INTER_NEAREST),
        "resize_linear_f05": lambda: cv2.resize(bgr, None, fx=0.5, fy=0.5),
        "resize_f32_area_to256": lambda: cv2.resize(gf, (256, 144), interpolation=cv2.INTER_AREA),
        "warp_affine_u8c3_4k": lambda: cv2.warpAffine(bgr, m, (3840, 2160)),
        "warp_affine_u8c1_4k": lambda: cv2.warpAffine(gray, m, (3840, 2160)),
        "warp_affine_f32c1_4k": lambda: cv2.warpAffine(gf, m, (3840, 2160), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE),
        "gauss_u8c3_k5": lambda: cv2.GaussianBlur(bgr, (5, 5), 0),
        "gauss_u8c3_s6": lambda: cv2.GaussianBlur(bgr, (0, 0), 6),
        "gauss_f32c1_s1": lambda: cv2.GaussianBlur(gf, (0, 0), 1.0),
        "gauss_f32c3_s1": lambda: cv2.GaussianBlur(bf, (0, 0), 1.0),
        "erode_f32_k5": lambda: cv2.erode(gf, np.ones((5, 5), np.uint8)),
        "copy_make_border_u8c3": lambda: cv2.copyMakeBorder(bgr, 8, 8, 8, 8, cv2.BORDER_REPLICATE),
        "transform_f32c3": lambda: cv2.transform(bf, y2b),
        "filter2d_f32c1_k9": lambda: cv2.filter2D(gf, -1, k9),
    }

    def best(f, reps=5):
        f()
        t = []
        for _ in range(reps):
            t0 = time.perf_counter()
            f()
            t.append((time.perf_counter() - t0) * 1e3)
        return round(min(t), 2)

    res = {}
    n0 = cv2.getNumThreads()
    for threads, key in ((n0, "cv2_ms"), (1, "cv2_1thread_ms")):
        cv2.setNumThreads(threads)
        for k, f in ops.items():
            res.setdefault(k, {})[key] = best(f)
    cv2.setNumThreads(n0)
    return res, n0


def main():
    args = sys.argv[1:]
    filt = args[args.index("--filter") + 1] if "--filter" in args else None
    if "--no-dump" not in args or not os.path.exists(os.path.join(OUT, "manifest.json")):
        run([sys.executable, os.path.join(HERE, "vision_dump.py"), OUT])
    run(["cargo", "build", "--release", "-q", "-p", "mm-vision", "--bin", "vision_parity"], cwd=NATIVE)
    exe = os.path.join(NATIVE, "target", "release", "vision_parity")
    man = json.load(open(os.path.join(OUT, "manifest.json")))
    args_by_case = {c["name"]: c["args"] for c in man["cases"]}
    lines = run([exe, "check", OUT] + ([filt] if filt else [])).splitlines()
    by_fn, failures, approx = {}, [], []
    for line in lines:
        d = json.loads(line)
        if "summary" in d:
            continue
        fn = d["fn"]
        s = by_fn.setdefault(fn, {"cases": 0, "exact": 0, "within_tolerance": 0, "failed": 0, "differing_elements": 0, "max_abs_diff": 0.0})
        s["cases"] += 1
        if "error" in d:
            s["failed"] += 1
            failures.append(d)
            continue
        s["differing_elements"] += d["differing"]
        s["max_abs_diff"] = max(s["max_abs_diff"], d["max_abs_diff"])
        if d["pass"]:
            s["exact"] += 1
            continue
        a = args_by_case.get(d["case"], {})
        if fn == "filter2d" and a.get("kw", 0) * a.get("kh", 0) >= 50 and d["max_abs_diff"] <= DFT_TOL:
            s["within_tolerance"] += 1
            approx.append({"case": d["case"], "differing": d["differing"], "of": d["of"], "max_abs_diff": d["max_abs_diff"], "why": "cv2 uses a DFT for >= 50 taps"})
        else:
            s["failed"] += 1
            failures.append(d)
    out = {"cv2": man["cv2"], "functions": by_fn, "failures": failures[:50], "tolerance_cases": approx}
    if "--bench" in args:
        rust = json.loads(run([exe, "bench", OUT]))
        cv, nthreads = bench_cv2()
        out["bench_4k_ms"] = {k: {"rust_ms": rust.get(k), **v} for k, v in cv.items()}
        out["bench_note"] = f"best of 5; cv2 default uses {nthreads} threads; Rust splits rows over all cores (mm_vision::set_num_threads), plain code, no SIMD intrinsics"
    print(json.dumps(out, indent=1))
    ok = not failures and sum(s["cases"] for s in by_fn.values()) > 0
    print(json.dumps({"pass": ok}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
