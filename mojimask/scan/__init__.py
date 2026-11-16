"""The scan: from a clip's detections and tracks to people, covers and the questions the app asks.

newalgo.py runs it (pipeline.py and the app call newalgo.run); the rules that settle questions are blips, overlaps,
same, cover and attach, with mojimask/facts.py. Modules import each other and the engine as
`from mojimask.scan import graph as G`, `from mojimask import track as T`. Modules with a command line run from the
repo root as `python -m mojimask.scan.<module> ...`. The benchmarks are in tools/bench.
"""
