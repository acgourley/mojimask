"""mm-motion against Python's cv2 motion, judged by quality (not equality; native/docs/porting.md, "write our own motion"):
runs tools/motion/known_motion.py (camera, known transforms), tools/motion/heads.py (head following) and
tools/motion/jazz_mats.py (real jazz and jazz_4k, speed), and prints their summaries as one JSON and a pass/fail line.
The decisions benchmark is separate (motion_bench.py, ~2 minutes).
    cargo build --release -p mm-motion && python native/tools/parity/motion_check.py
"""
import json
import os
import subprocess
import sys

# the main checkout (a worktree has no venv or models of its own): the parent of git's common directory
MAIN_CHECKOUT = os.path.dirname(subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse",
                                                "--path-format=absolute", "--git-common-dir"],
                                               capture_output=True, text=True).stdout.strip())

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(NATIVE)
PY = os.path.join(MAIN_CHECKOUT, ".venv/bin/python")
OUT = os.path.join(ROOT, "out", "native", "motion")


def main():
    lines, ok = {}, True
    for tool, js in (("known_motion", "known_motion.json"), ("heads", "heads.json"), ("jazz_mats", "jazz_mats.json")):
        r = subprocess.run([PY, os.path.join(NATIVE, "tools", "motion", f"{tool}.py")], capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": ROOT})
        last = (r.stdout.strip().splitlines() or ["FAIL (no output) " + r.stderr[-500:]])[-1]
        ok &= last.startswith("PASS")
        d = json.load(open(os.path.join(OUT, js)))
        lines[tool] = {"line": last, "summary": d.get("summary", d)}
    print(json.dumps(lines, indent=1, default=str))
    print(f"{'PASS' if ok else 'FAIL'} motion_check: " + "; ".join(v["line"] for v in lines.values()))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
