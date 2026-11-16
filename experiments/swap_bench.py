"""The decisions bench with other face models swapped in, in a scratch copy (2026-11-14).

(Written while the app used SCRFD + ArcFace; since the switch, "base" is RetinaFace + SFace and "insightface" the old
pair. Its results were taken before the switch, patching reid.scrfd/arcface.)

Runs tools/bench/decision_bench.py on jazz with reid's SCRFD and/or ArcFace replaced in this process only
(no app or scan code changed), in out/bench_swap/<variant>/ instead of out/bench/: its own copy of jazz's cached
detections, its own lock and history file, links to the real out/truth and test_clips.

  base         today's models (the baseline, same code and copy)
  insightface  the old pair, SCRFD + ArcFace, with ArcFace's thresholds in graph.py
  yunet        YuNet 2023mar for landmarks (crop at 160 px on its 640 canvas), ArcFace kept
  retina       the app's own RetinaFace-R50 (detection model, MIT) for landmarks, its landmark head decoded
  ...-mapped   the same with YuNet's scores mapped onto SCRFD's scale (out/face_compare/yunet_to_scrfd.json, from
               experiments/face_score_calibrate.py), so the scan's fixed face-check thresholds mean the same
  yunet-sface  YuNet + SFace 2021dec, graph.py's face thresholds moved to SFace's scale (same error rates on the
               jazz census groups as the app's 0.55 / 0.40 / 0.25 under ArcFace: experiments/face_model_compare.py)
               (questions.py's ranking centres faces at SFace's 0.49 in every variant: left alone)

    .venv/bin/python experiments/swap_bench.py VARIANT [CLIP (default jazz)] [decision_bench options, e.g. --sheets]

A clip with no out/bench/<clip>/ is detected once (by whichever variant runs first; detection doesn't use these
models) and the detections kept in out/bench_swap/_dets/<clip>/ for the other variants.
"""
import os
import pickle
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "tools", "bench")
sys.path[:0] = [ROOT, HERE, os.path.join(ROOT, "experiments")]
SFACE_THRESHOLDS = dict(SAME_FACE=0.66, ASK_FACE=0.57, DIFF_FACE=0.33)  # (graph.py's own since the switch)
ARCFACE_THRESHOLDS = dict(SAME_FACE=0.55, ASK_FACE=0.4, DIFF_FACE=0.25)
KEEP = ("frames", "dets", "meta.json", "mats_v2.npy", "evidence_dets.pkl", "preview.jpg")  # detection, not decisions


def det_source(clip):
    for d in (os.path.join(ROOT, "out", "bench", clip), os.path.join(ROOT, "out", "bench_swap", "_dets", clip)):
        if os.path.exists(os.path.join(d, "dets", "pose_head.json")):
            return d
    return None


def scratch(variant, clip):
    root = os.path.join(ROOT, "out", "bench_swap", variant)
    proj = os.path.join(root, "out", "bench", clip)
    shutil.rmtree(proj, ignore_errors=True)
    os.makedirs(proj)
    src = det_source(clip) or proj
    for k in KEEP:
        s = os.path.join(src, k)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(proj, k))
        elif os.path.exists(s):
            shutil.copy2(s, proj)
    os.makedirs(os.path.join(proj, "proto"))
    pickle.dump({}, open(os.path.join(proj, "proto", "faces.pkl"), "wb"))  # never an older run's face cache
    for link, target in (("out/truth", "out/truth"), ("out/probe", "out/probe"), ("out/projects", "out/projects"),
                         ("test_clips", "test_clips")):
        p = os.path.join(root, link)
        if not os.path.lexists(p) and os.path.exists(os.path.join(ROOT, target)):
            os.symlink(os.path.join(ROOT, target), p)
    return root


def main():
    variant = sys.argv[1]
    # VARIANT: base (the app's models), insightface (the old SCRFD + ArcFace, with ArcFace's thresholds), or
    # <yunet|retina>[-sface][-mapped] (landmarks from YuNet / RetinaFace-on-the-CPU-path, likeness SFace or ArcFace)
    parts = variant.split("-")
    assert variant in ("base", "insightface") or (parts[0] in ("yunet", "retina") and set(parts[1:]) <= {"sface", "mapped"}), variant
    import json
    import numpy as np
    from mojimask import reid
    from mojimask.scan import graph
    import face_model_compare as FMC
    import insightface_models as IF
    if variant != "base":  # everything but base starts from the old pair, as these experiments did
        reid.landmarks = lambda img, size=320, thresh=0.3, iou=0.4, engine=None: IF.scrfd(img, thresh=thresh, iou=iou)
        reid.embed = lambda aligned: IF.arcface(aligned).flatten()
        for k, v in ARCFACE_THRESHOLDS.items():
            setattr(graph, k, v)
        print("note: questions.py's face ranking centre stays at SFace's 0.49 in this variant", file=sys.stderr)
    if parts[0] in ("yunet", "retina"):
        det = parts[0]
        run = FMC.yunet if det == "yunet" else FMC.retina
        m = json.load(open(os.path.join(ROOT, "out", "face_compare", f"{det}_to_scrfd.json"))) if "mapped" in parts else None

        def landmarks(img, size=320, thresh=0.3, iou=0.4, engine=None):
            b, k = run(img, thresh=thresh, iou=iou)
            if m is not None and len(b):
                b = b.copy()
                b[:, 4] = np.interp(b[:, 4], m[det], m["scrfd"])
            return b, k
        reid.landmarks = landmarks
    if "sface" in parts:
        reid.embed = lambda aligned: FMC.sface(aligned)
        for k, v in SFACE_THRESHOLDS.items():
            setattr(graph, k, v)
    import decision_bench as DB
    rest = sys.argv[2:]
    clip = rest.pop(0) if rest and not rest[0].startswith("-") else "jazz"
    fresh = det_source(clip) is None
    root = scratch(variant, clip)
    DB.ROOT = root
    DB.HISTORY = os.path.join(root, "history.jsonl")
    DB.main([clip, "--note", f"swap: {variant}"] + rest)  # (its code fingerprint still reads DB.CODE_ROOT, the checkout)
    if fresh:  # keep this clip's detections for the other variants
        proj, keep = os.path.join(root, "out", "bench", clip), os.path.join(ROOT, "out", "bench_swap", "_dets", clip)
        os.makedirs(keep, exist_ok=True)
        for k in KEEP:
            s = os.path.join(proj, k)
            if os.path.isdir(s):
                shutil.copytree(s, os.path.join(keep, k), dirs_exist_ok=True)
            elif os.path.exists(s):
                shutil.copy2(s, keep)


if __name__ == "__main__":
    main()
