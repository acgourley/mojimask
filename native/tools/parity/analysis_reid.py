"""mm-analysis reid / faces / appearance against Python on jazz: the same detections through both.

    PYTHONPATH=ROOT .venv/bin/python native/tools/parity/analysis_reid.py [--every N] [--coreml] [--no-build]

Detections: every tracklet detection of jazz (scan/faces.measure's input) on frames t % N == 0 (default 4; 1 = all
2,723). Per detection: reid.face_crop (pixels), reid.landmarks (boxes, landmarks), faces.nearest and the numbers
(score, yaw, eyes, px, frontal), the alignment matrix, reid.align (pixels), reid.embed (raw and normalised), and
faces.face end to end (found, frontal). Appearance: OSNet crops (pixels) and embeddings, in batches as
scan/questions.py makes them.

Default: the CPU EP on both sides; pass = every crop and aligned face identical, landmarks and raw embeddings
identical, every decision identical, normalised embeddings within 1e-6 (numpy normalises with Accelerate's sdot).
--coreml: onnx_core's default plan on both sides (landmarks on the ANE, SFace on the GPU; OSNet stays on the CPU):
reports the differences, pass = the same decisions. Prints JSON; the last line is {"pass": ...}.
"""
import json
import os
import sys

import numpy as np

from analysis_common import (CACHE, MODELS, build, ep_arg, project, run_rust, save_frames, stats, timed,
                             tracklet_dets, workdir)


def main():
    every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 4
    ep = ep_arg()
    if "--no-build" not in sys.argv:
        build()
    import cv2
    from mojimask import appearance as AP
    from mojimask import pose_util as PU
    from mojimask import reid
    from mojimask.scan import faces as FA
    if ep == "cpu":
        reid.LANDMARK_ENGINE = "cpu"
        reid.ENGINE = "cpu"
    proj = project()
    dets, tl = tracklet_dets(every)
    d = workdir("reid", ep)
    save_frames(d, [t for _, t, _, _ in dets])
    imgs = {}
    py, times = {}, {}
    # warm-up (sessions open)
    img0 = cv2.imread(f"{proj}/frames/{dets[0][1]:05d}.jpg")
    FA.face(img0, dets[0][2])
    reid.embed(np.zeros((112, 112, 3), np.uint8))
    for i, t, c, _ in dets:
        img = imgs.setdefault(t, cv2.imread(f"{proj}/frames/{t:05d}.jpg"))
        f = timed(times, "face", FA.face, img, c)
        cr = timed(times, "face_crop", reid.face_crop, img, c)
        e = {"face_found": f is not None, "face_frontal": FA.frontal(f) if f is not None else None}
        if cr is None:
            py[i] = {**e, "crop": None}
            continue
        crop, k, x0, y0 = cr
        np.save(f"{d}/py/crop_{i}.npy", crop)
        boxes, kps = timed(times, "landmarks", reid.landmarks, crop)
        j = FA.nearest(boxes, k, x0, y0, c)
        e.update(k=k, x0=x0, y0=y0, boxes=boxes, kps=kps, nearest=j)
        if j is not None:
            b, kp = boxes[j], kps[j]
            ed = float(np.hypot(*(kp[1] - kp[0]))) + 1e-6
            e.update(score=float(b[4]), yaw=float((kp[2][0] - (kp[0][0] + kp[1][0]) / 2) / ed),
                     eyes=float(ed / max(1.0, b[2] - b[0])), px=float((b[3] - b[1]) / k))
            e["frontal"] = FA.frontal({"score": e["score"], "yaw": e["yaw"], "px": e["px"]})
            e["matrix"] = reid.similarity(np.asarray(kp, np.float64), reid.ALIGN_DST.astype(np.float64))
            al = timed(times, "align", reid.align, crop, kp)
            np.save(f"{d}/py/aligned_{i}.npy", al)
            raw = timed(times, "embed", reid.embed, al).flatten()
            e["emb_raw"] = raw
            e["emb"] = (raw / np.linalg.norm(raw)).astype(np.float32)
        py[i] = e
    rust, wall = run_rust("reid", d, {"ep": ep, "models": MODELS, "cache": CACHE,
                                      "dets": [[i, t, [float(v) for v in c]] for i, t, c, _ in dets]})
    rs = {e["i"]: e for e in rust["dets"]}
    res = {"detections": len(dets), "ep": ep}
    n = lambda key: sum(1 for i in py if key(py[i], rs[i]))  # noqa: E731
    res["crop_none_mismatch"] = n(lambda p, r: (p.get("crop", 1) is None) != (r.get("crop", 1) is None))
    both = [i for i in py if py[i].get("crop", 1) is not None and rs[i].get("crop", 1) is not None]
    res["crops_compared"] = len(both)
    res["crop_pixels_differing"] = int(sum(max(rs[i]["crop_diff"], 0) for i in both))
    res["crop_shape_mismatch"] = sum(1 for i in both if rs[i]["crop_diff"] < 0)
    res["k_x0_y0_mismatch"] = sum(1 for i in both if (py[i]["k"], py[i]["x0"], py[i]["y0"]) != (rs[i]["k"], rs[i]["x0"], rs[i]["y0"]))
    nb_mis, box_d, kp_d = 0, 0.0, 0.0
    for i in both:
        pb, rb = np.asarray(py[i]["boxes"], np.float32).reshape(-1, 5), np.asarray(rs[i]["boxes"], np.float32).reshape(-1, 5)
        if pb.shape != rb.shape:
            nb_mis += 1
            continue
        if len(pb):
            box_d = max(box_d, float(np.abs(pb - rb).max()))
            kp_d = max(kp_d, float(np.abs(np.asarray(py[i]["kps"], np.float32).reshape(-1, 5, 2)
                                          - np.asarray(rs[i]["kps"], np.float32).reshape(-1, 5, 2)).max()))
    res.update(face_count_mismatch=nb_mis, box_max_abs=box_d, kps_max_abs=kp_d)
    res["nearest_mismatch"] = sum(1 for i in both if py[i]["nearest"] != rs[i]["nearest"])
    found = [i for i in both if py[i]["nearest"] is not None and rs[i]["nearest"] is not None]
    res["faces_found"] = len(found)
    for key in ("score", "yaw", "eyes", "px"):
        res[f"{key}_max_abs"] = max([abs(py[i][key] - rs[i][key]) for i in found] or [0.0])
    res["frontal_mismatch"] = sum(1 for i in found if py[i]["frontal"] != rs[i]["frontal"])
    res["frontal_count"] = sum(1 for i in found if py[i]["frontal"])
    res["matrix_max_abs"] = max([float(np.abs(np.asarray(py[i]["matrix"]).ravel() - np.asarray(rs[i]["matrix"])).max()) for i in found] or [0.0])
    res["aligned_pixels_differing"] = int(sum(max(rs[i]["aligned_diff"], 0) for i in found))
    res["aligned_shape_mismatch"] = sum(1 for i in found if rs[i]["aligned_diff"] < 0)
    er = [float(np.abs(np.asarray(rs[i]["emb_raw"], np.float32) - py[i]["emb_raw"]).max()) for i in found]
    en = [float(np.abs(np.asarray(rs[i]["emb"], np.float32) - py[i]["emb"]).max()) for i in found]
    cs = [float(np.dot(np.asarray(rs[i]["emb"], np.float64), py[i]["emb"].astype(np.float64))) for i in found]
    res.update(emb_raw_max_abs=max(er or [0.0]), emb_max_abs=max(en or [0.0]), emb_cosine_min=min(cs or [1.0]),
               emb_identical=sum(1 for v in en if v == 0.0))
    res["face_found_mismatch"] = n(lambda p, r: p["face_found"] != r["face_found"])
    res["face_frontal_mismatch"] = n(lambda p, r: p["face_frontal"] != r["face_frontal"])

    # ---- appearance (OSNet)
    with open(f"{proj}/dets/pose_full.json") as f:
        poses = json.load(f)["frames"]
    batches, pyb = [], []
    for k, v in list(tl.items()):
        ts = sorted(v["obs"])
        if len(ts) < 6:
            continue
        rows, crops = [], []
        for t in ts[::max(1, len(ts) // 8)]:
            sk = PU.pose_for_head(poses[t], v["obs"][t])
            if sk is None:
                continue
            img = imgs.get(t)
            if img is None:
                img = imgs[t] = cv2.imread(f"{proj}/frames/{t:05d}.jpg")
            box = AP.body_box(v["obs"][t], sk)
            cr = timed(times, "body_crop", AP.crop, img, box)
            rows.append([t, *[float(x) for x in box]])
            if cr is not None:
                np.save(f"{d}/py/body_{len(batches)}_{len(rows) - 1}.npy", cr)
                crops.append(cr)
        if len(crops) >= 2:
            if not pyb:
                AP.reid_embed(crops)
            batches.append(rows)
            pyb.append(timed(times, "reid_embed", AP.reid_embed, crops))
        if len(batches) >= 30:
            break
    save_frames(d, [r[0] for b in batches for r in b])
    ra, wall_a = run_rust("appearance", d, {"ep": ep, "models": MODELS, "cache": CACHE, "batches": batches})
    diffs = [x for b in ra["batches"] for x in b["crop_diffs"]]
    res["body_crops"] = len(diffs)
    res["body_crop_pixels_differing"] = int(sum(x for x in diffs if x > 0))
    res["body_crop_mismatch"] = sum(1 for x in diffs if x < 0)
    ad = [float(np.abs(np.asarray(r["emb"], np.float32) - p).max()) for r, p in zip(ra["batches"], pyb)]
    ac = [float((np.asarray(r["emb"], np.float64) * p.astype(np.float64)).sum(1).min()) for r, p in zip(ra["batches"], pyb)]
    res.update(osnet_batches=len(pyb), osnet_max_abs=max(ad or [0.0]), osnet_cosine_min=min(ac or [1.0]),
               osnet_batches_identical=sum(1 for v in ad if v == 0.0))

    res["speed"] = {k: {"python": stats(times.get(k)), "rust": stats(rust["times"].get(k) or ra["times"].get(k.replace("body_", "")))}
                    for k in ("face", "face_crop", "landmarks", "align", "embed", "body_crop", "reid_embed")}
    if ep == "cpu":
        ok = (res["crop_none_mismatch"] == 0 and res["crop_pixels_differing"] == 0 and res["crop_shape_mismatch"] == 0
              and res["k_x0_y0_mismatch"] == 0 and res["face_count_mismatch"] == 0 and res["box_max_abs"] == 0
              and res["kps_max_abs"] == 0 and res["nearest_mismatch"] == 0 and res["score_max_abs"] == 0
              and res["yaw_max_abs"] == 0 and res["eyes_max_abs"] == 0 and res["px_max_abs"] == 0
              and res["frontal_mismatch"] == 0 and res["aligned_pixels_differing"] == 0
              and res["emb_raw_max_abs"] == 0 and res["emb_max_abs"] <= 1e-6 and res["face_found_mismatch"] == 0
              and res["face_frontal_mismatch"] == 0 and res["body_crop_pixels_differing"] == 0
              and res["body_crop_mismatch"] == 0 and res["osnet_max_abs"] <= 1e-6)
    else:
        ok = res["face_found_mismatch"] == 0 and res["face_frontal_mismatch"] == 0 and res["nearest_mismatch"] == 0
    print(json.dumps(res, indent=1))
    print(json.dumps({"pass": bool(ok)}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
