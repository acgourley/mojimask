"""Seek parity: every slot of every parity clip reached by a seek (Decoder::seek + next) must be the frame decoding
forward gives for that slot: same picture (luma hash) and same source pts. No Python involved: forward decoding is
checked against video_io by media_decode.py. Also reports seek latency (seek + next to the slot's frame, 100 seeks
at spread-out slots) on jazz_4k, the clip with the costliest frames.

Fixed 2026-11-17 (decode.rs, Decoder::seek: frames identified by labels, not the decoder's pts); before, random
slots on jazz (0-3 per run) and the last 2-4 slots of jazz_vfr and jazz_hlg10 were wrong."""
import sys

from _media import build, clips, report, rust

build()
rows, bad = {}, 0
for c in clips(extra=False):
    r = rust("seek-all", c)
    rows[c] = [m["t"] for m in r["seek_mismatches"]]
    bad += len(rows[c])
lat = {c: rust("seek-bench", c, 100) for c in clips(extra=False) if "4k" in c}
sys.exit(report("media_seek", bad == 0, wrong_slots=rows, latency_ms=lat))
