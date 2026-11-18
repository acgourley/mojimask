"""mm-analysis pose_util / head_pose against Python on jazz's skeletons (dets/pose_full.json).

    PYTHONPATH=ROOT .venv/bin/python native/tools/parity/analysis_pose.py [--no-build]

Every skeleton of every frame: pose_util.head, facing, head_box, pose_head (s = 40), head_pose.raw_angles; every
tracklet detection: pose_util.pose_for_head; head_pose.track_angles over all covers (every detection, some as
split pieces); head_pose.mirror_for on every listed emoji and a few others. Exact (pass = every number equal).
Prints JSON; the last line is {"pass": ...}.
"""
import json
import sys

from analysis_common import build, project, run_rust, stats, timed, tracklet_dets, workdir


def main():
    if "--no-build" not in sys.argv:
        build()
    from mojimask import head_pose as HP
    from mojimask import pose_util as PU
    proj = project()
    with open(f"{proj}/dets/pose_full.json") as f:
        poses = json.load(f)["frames"]
    d = workdir("pose", "cpu")
    times = {}
    rows = []
    for fr in poses:
        for p in fr:
            h = timed(times, "head", PU.head, p)
            hb = PU.head_box(p)
            xy, q = timed(times, "pose_head", PU.pose_head, p, 40.0)
            a = timed(times, "raw_angles", HP.raw_angles, p)
            rows.append({"head": list(h) if h else None, "facing": PU.facing(p), "head_box": hb,
                         "pose_head": [float(xy[0]), float(xy[1]), float(q)], "raw": [float(v) for v in a]})
    dets, tl = tracklet_dets(1)
    pfh_q = [[t, [float(v) for v in c]] for _, t, c, _ in dets]
    pfh = []
    for t, c in pfh_q:
        r = timed(times, "pose_for_head", PU.pose_for_head, poses[t], c)
        pfh.append(None if r is None else next(i for i, x in enumerate(poses[t]) if x is r))
    per_frame = [[] for _ in poses]
    for _, t, c, k in dets:
        per_frame[t].append([str(k), [float(v) for v in c], "obs"] + ([f"{k}~2"] if k % 4 == 0 else []))
    ang = timed(times, "track_angles", HP.track_angles, [[tuple(x) for x in f] for f in per_frame], poses)
    ang = {str(k): {str(t): [float(a[0]), float(a[1]), int(a[2])] for t, a in v.items()} for k, v in ang.items()}
    emoji = list(HP.FACING) + ["😀", "🦉", "🍕", "🐈", "a", "🐈‍⬛x"]
    mirror_q = [[e, s] for e in emoji for s in (-1, 0, 1)]
    mirror = [HP.mirror_for(e, s) for e, s in mirror_q]
    rust, _ = run_rust("pose", d, {"poses": poses, "pose_for_head": pfh_q, "per_frame": per_frame, "mirror": mirror_q})
    res = {"skeletons": len(rows)}
    res["rows_mismatch"] = sum(1 for a, b in zip(rows, rust["rows"]) if a != b)
    bad = [(a, b) for a, b in zip(rows, rust["rows"]) if a != b]
    if bad:
        res["first_row_mismatch"] = {"python": bad[0][0], "rust": bad[0][1]}
    res["pose_for_head_queries"] = len(pfh)
    res["pose_for_head_mismatch"] = sum(1 for a, b in zip(pfh, rust["pose_for_head"]) if a != b)
    res["track_angles_pieces"] = len(ang)
    res["track_angles_values"] = sum(len(v) for v in ang.values())
    res["track_angles_mismatch"] = sum(1 for k in ang for t in ang[k] if rust["track_angles"].get(k, {}).get(t) != ang[k][t]) \
        + abs(len(ang) - len(rust["track_angles"]))
    res["mirror_mismatch"] = sum(1 for a, b in zip(mirror, rust["mirror"]) if a != b)
    rt = rust["times"]
    n = len(rows)
    res["speed"] = {"per skeleton (head + pose_head + raw_angles + facing + head_box)": {
        "python_ms": round(sum(sum(times[k]) for k in ("head", "pose_head", "raw_angles")) / n, 4),
        "rust_ms": round(rt["rows_ms"] / n, 4)},
        "pose_for_head": {"python": stats(times["pose_for_head"]), "rust_mean_ms": rt["pose_for_head_ms"] / len(pfh)},
        "track_angles (whole clip)": {"python_ms": round(times["track_angles"][0], 2), "rust_ms": round(rt["track_angles_ms"], 2)}}
    ok = res["rows_mismatch"] == 0 and res["pose_for_head_mismatch"] == 0 and res["track_angles_mismatch"] == 0 \
        and res["mirror_mismatch"] == 0
    print(json.dumps(res, indent=1))
    print(json.dumps({"pass": bool(ok)}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
