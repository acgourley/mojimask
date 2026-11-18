"""mm-analysis depth against Python's depth.py on jazz (Depth Anything V2 Small).

    PYTHONPATH=ROOT .venv/bin/python native/tools/parity/analysis_depth.py [--coreml] [--every N] [--no-build]

1. The model (depth._Onnx): every Nth frame (default 8) through both: the input tensor (bit-identical) and the
   output map (max |diff|). CPU EP by default; --coreml runs both on the "gpu" engine (CoreML ML Program).
2. The cache's numbers (depth.compute's loop): each frame's 8-bit picture (identical), percentiles and the fit.
3. Depth over the cached movie (the movie decoded by Python, handed to Rust as luma frames, since decoding is
   mm-media's): Depth.maps (identical), Depth.full, near for every tracklet detection, gap for 300 tracklet pairs,
   face_at on random cases with ties, and Occlusion: map (identical), nearness and explain (every decision and
   number) on every frame with every detection as a cover.
Prints JSON; the last line is {"pass": ...}. With --coreml only the model step runs (the rest has no model).
"""
import json
import os
import sys

import numpy as np

from analysis_common import (CACHE, DEPTH_MODEL, MODELS, build, ep_arg, project, run_rust, save_frames, stats, timed,
                             tracklet_dets, workdir)


class _Rec:
    """Wraps an onnxruntime session to keep the last input."""

    def __init__(self, s):
        self.s, self.x = s, None

    def get_inputs(self):
        return self.s.get_inputs()

    def run(self, names, feeds):
        self.x = next(iter(feeds.values()))
        return self.s.run(names, feeds)


def main():
    ep = ep_arg()
    every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 8
    if "--no-build" not in sys.argv:
        build()
    import cv2
    from mojimask import depth as DP
    from mojimask import onnx_core
    proj = project()
    nframes = len(os.listdir(os.path.join(proj, "frames")))
    d = workdir("depth", ep)
    res, times = {"ep": ep}, {}

    # ---- 1. the model
    ts = list(range(0, nframes, every))
    save_frames(d, ts)
    run = DP._Onnx(DEPTH_MODEL)
    maps = {}
    for i, t in enumerate(ts):
        img = cv2.imread(f"{proj}/frames/{t:05d}.jpg")
        h, w = img.shape[:2]
        k = DP.SIDE / max(h, w)
        hw = (max(14, int(round(h * k / 14)) * 14), max(14, int(round(w * k / 14)) * 14))
        if hw not in run.sessions:
            run.sessions[hw] = _Rec(onnx_core.session(DEPTH_MODEL, "cpu" if ep == "cpu" else "gpu",
                                                      dims={"batch_size": 1, "height": hw[0], "width": hw[1]}))
        if i == 0:
            run(img)
        m = timed(times, "depth_model", run, img).astype(np.float32)
        np.save(f"{d}/py/din_{t}.npy", run.sessions[hw].x)
        np.save(f"{d}/py/depth_{t}.npy", m)
        maps[t] = m
    rm, _ = run_rust("depth-model", d, {"ep": ep, "models": MODELS, "cache": CACHE, "model": DEPTH_MODEL, "frames": ts})
    res["model_frames"] = len(ts)
    res["model_input_values_differing"] = int(sum(f["input_diff"] for f in rm["frames"]))
    dm = [float(np.abs(np.load(f"{d}/rust/depth_{t}.npy") - maps[t]).max()) for t in ts]
    rng = [float(maps[t].max() - maps[t].min()) for t in ts]
    res["model_max_abs"] = max(dm)
    res["model_max_abs_over_range"] = max(a / r for a, r in zip(dm, rng))
    speed = {"depth_model": {"python": stats(times["depth_model"]), "rust": stats(rm["times"])}}
    if ep != "cpu":
        res["speed"] = speed
        print(json.dumps(res, indent=1))
        ok = res["model_input_values_differing"] == 0
        print(json.dumps({"pass": ok}))
        return 0 if ok else 1

    # ---- 2. the cache's numbers (depth.compute's per-frame lines, verbatim)
    fit_cases, fit_py = [], []
    luma = [y for _, y in DP._luma(os.path.join(proj, "dets", "depth_anything-small.mp4"))]
    for t in ts:
        n = maps[t]
        h, wd = n.shape
        H2, W2 = h + h % 2, wd + wd % 2
        lo, hi = float(n.min()), float(n.max())
        q = np.round((n - lo) / max(hi - lo, 1e-9) * 255).astype(np.uint8)
        q = cv2.copyMakeBorder(q, 0, H2 - h, 0, W2 - wd, cv2.BORDER_REPLICATE)
        np.save(f"{d}/py/q_{t}.npy", q)
        p10, p50, p90 = np.percentile(n, [10, 50, 90])
        maps[t] = (maps[t], [float(p10), float(p50), float(p90)], lo, hi)
        x = luma[t][:h:4, :wd:4].astype(np.float64).ravel()
        y = n[::4, ::4].astype(np.float64).ravel()
        fit_cases.append({"x": x.tolist(), "y": y.tolist()})
        fit_py.append([float(v) for v in np.polyfit(x, y, 1)])
    re, _ = run_rust("depth-encode", d, {"frames": ts, "fit_cases": fit_cases})
    res["encode_q_values_differing"] = int(sum(max(f["q_diff"], 0) for f in re["frames"]))
    res["encode_q_shape_mismatch"] = sum(1 for f in re["frames"] if f["q_diff"] < 0)
    res["encode_pct_max_abs"] = max(float(np.abs(np.array(f["pct"]) - maps[f["t"]][1]).max()) for f in re["frames"])
    res["encode_lo_hi_mismatch"] = sum(1 for f in re["frames"] if (f["lo"], f["hi"]) != (maps[f["t"]][2], maps[f["t"]][3]))
    res["fit_max_rel"] = max(float(np.abs((np.array(r) - p) / np.maximum(np.abs(p), 1e-12)).max()) for r, p in zip(re["fits"], fit_py))

    # ---- 3. Depth over the cached movie, Occlusion
    os.environ["MOJIMASK_DEPTH"] = "anything-small"
    D = timed(times, "depth_maps", DP.Depth, proj, "anything-small")
    np.save(f"{d}/py/luma.npy", np.stack(luma))
    dets, tl = tracklet_dets(1)
    near_q = [[t, [float(v) for v in c]] for _, t, c, _ in dets]
    near_py = [timed(times, "near", D.near, t, np.array(c)) for t, c in near_q]
    keys = sorted(tl)
    r = np.random.default_rng(7)
    pairs = [(keys[a], keys[b]) for a, b in r.integers(0, len(keys), (300, 2)) if a != b]
    gaps_q = [[{str(t): [float(v) for v in c] for t, c in tl[a]["obs"].items()},
               {str(t): [float(v) for v in c] for t, c in tl[b]["obs"].items()}] for a, b in pairs]
    gaps_py = [D.gap(tl[a]["obs"], tl[b]["obs"]) for a, b in pairs]
    fa_q, fa_py = [], []
    for i in range(400):
        n = int(r.integers(1, 40))
        tts = np.sort(r.choice(np.arange(0, 127), n, replace=False)) if i % 2 else np.sort(r.integers(0, 60, n))
        tts = np.unique(tts)
        v = np.round(r.standard_normal(len(tts)) * 2, 1 if i % 3 == 0 else 6)  # rounded: ties in the values too
        t = int(r.integers(-5, 130))
        fa_q.append([tts.tolist(), v.tolist(), t])
        fa_py.append(DP.face_at(tts, v, t))
    obs = {str(k): {int(t): [float(x) for x in c] for t, c in v["obs"].items()} for k, v in tl.items()}
    occ = DP.Occlusion(D, obs)
    per_t = {}
    for _, t, c, k in dets:
        kind = "coast" if k % 3 == 0 else ("manual" if k % 11 == 0 else "obs")
        it = [str(k), [float(x) for x in c], kind] + ([f"{k}~1"] if k % 5 == 0 else [])
        per_t.setdefault(t, []).append(it)
    H, W = cv2.imread(f"{proj}/frames/00000.jpg").shape[:2]
    scale = 1.5
    occ_frames = [[t, per_t[t]] for t in sorted(per_t)]
    explain_py, nearness_py = [], []
    for t, items in occ_frames:
        its = [(x[0], x[1], x[2], *x[3:]) for x in items]
        explain_py.append(timed(times, "explain", occ.explain, t, its, 1.5, (H, W)))
        nearness_py.append([occ.nearness(t, x[3] if len(x) > 3 else x[0], x[1]) for x in its])
    occ_maps = [0, 51, 126]
    for t in occ_maps:
        np.save(f"{d}/py/occmap_{t}.npy", occ.map(t))
    full_frames = [0, 17, 64, 126]
    rd, _ = run_rust("depth-maps", d, {
        "sidecar": os.path.join(proj, "dets", "depth_anything-small.json"), "model_name": "anything-small",
        "frame_hw": [H, W], "full_frames": full_frames, "near": near_q, "gaps": gaps_q, "face_at": fa_q,
        "occ_obs": {k: {str(t): c for t, c in v.items()} for k, v in obs.items()}, "occ_scale": scale,
        "occ_frames": occ_frames, "occ_maps": occ_maps})
    rmaps = np.load(f"{d}/rust/maps.npy")
    res["maps_shape_match"] = rmaps.shape == D.maps.shape
    res["maps_values_differing"] = int((rmaps != D.maps).sum()) if res["maps_shape_match"] else -1
    res["maps_max_abs"] = float(np.abs(rmaps - D.maps).max()) if res["maps_shape_match"] else None
    res["full_values_differing"] = int(sum((np.load(f"{d}/rust/full_{t}.npy") != D.full(t)).sum() for t in full_frames))
    eq = lambda a, b: (a is None and (b is None or (isinstance(b, float) and np.isnan(b)))) or a == b  # noqa: E731
    res["near_queries"] = len(near_q)
    res["near_mismatch"] = sum(1 for a, b in zip(rd["near"], near_py) if not eq(a, b))
    res["gap_queries"] = len(gaps_q)
    res["gap_mismatch"] = sum(1 for a, b in zip(rd["gaps"], gaps_py) if not eq(a, b))
    res["face_at_cases"] = len(fa_q)
    res["face_at_mismatch"] = sum(1 for a, b in zip(rd["face_at"], fa_py) if not eq(a, b))
    res["occ_map_values_differing"] = int(sum((np.load(f"{d}/rust/occmap_{t}.npy") != occ.map(t)).sum() for t in occ_maps))
    covers = cuts = mism = nmis = 0
    first_bad = None
    for p, rr, npy_, (t, _) in zip(explain_py, rd["plans"], nearness_py, occ_frames):
        for a, b in zip(p, rr["explain"]):
            covers += 1
            cuts += a["cut"]
            a2 = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in a.items()}
            if a2 != b:
                mism += 1
                first_bad = first_bad or {"t": t, "python": a2, "rust": b}
        for a, b in zip(npy_, rr["nearness"]):
            nmis += not ((b is None and a == -np.inf) or a == b)
    res.update(occ_covers=covers, occ_cuts=cuts, occ_explain_mismatch=mism, occ_nearness_mismatch=nmis)
    if first_bad:
        res["occ_first_mismatch"] = first_bad
    speed["depth_maps (Depth init, 127 frames)"] = {"python": stats(times["depth_maps"]), "rust": {"ms": rd["times"]["maps_ms"]}}
    speed["near"] = {"python": stats(times["near"]), "rust": {"mean_ms": rd["times"]["near_ms"]}}
    speed["explain (one frame)"] = {"python": stats(times["explain"]), "rust": stats(rd["times"]["explain_ms"])}
    res["speed"] = speed
    ok = (res["model_input_values_differing"] == 0 and res["model_max_abs"] == 0 and res["encode_q_values_differing"] == 0
          and res["encode_q_shape_mismatch"] == 0 and res["encode_pct_max_abs"] == 0 and res["encode_lo_hi_mismatch"] == 0
          and res["fit_max_rel"] < 1e-9 and res["maps_values_differing"] == 0 and res["full_values_differing"] == 0
          and res["near_mismatch"] == 0 and res["gap_mismatch"] == 0 and res["face_at_mismatch"] == 0
          and res["occ_map_values_differing"] == 0 and res["occ_explain_mismatch"] == 0 and res["occ_nearness_mismatch"] == 0)
    print(json.dumps(res, indent=1, default=float))
    print(json.dumps({"pass": bool(ok)}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
