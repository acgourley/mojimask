"""Python reference export without covers: video_io frames (display) -> formats.fit_frame -> video_io.writer -> video_io.mux.
The same path as track.render with no covers drawn.   _py_export.py CLIP OUT PRESET FIT [START END]"""
import json, os, shutil, sys, tempfile
from mojimask import formats, video_io

clip, out, preset, fit = sys.argv[1:5]
start = float(sys.argv[5]) if len(sys.argv) > 5 else None
end = float(sys.argv[6]) if len(sys.argv) > 6 else None
meta = video_io.probe(clip)
o = formats.settings(meta, preset, fit, start, end)
if meta.get("image"):
    f = next(video_io.frames(clip, meta, color="display"))
    video_io.write_image(out, formats.fit_frame(f, o["width"], o["height"], o["fit"]))
    print(json.dumps(o)); sys.exit()
s, e = o["start"], o["end"]
d = tempfile.mkdtemp(prefix="mm-parity-")
try:
    tmp = os.path.join(d, "video.mp4")
    w = video_io.writer(tmp, o["width"], o["height"], o["fps"], o["bitrate"])
    for t, f in enumerate(video_io.frames(clip, meta, color="display")):
        if t >= e:
            break
        if t < s or (t - s) % o["step"]:
            continue
        w.write(formats.fit_frame(f, o["width"], o["height"], o["fit"]))
    w.close()
    video_io.mux(tmp, clip, out, start=s / meta["fps"])
finally:
    shutil.rmtree(d, ignore_errors=True)
print(json.dumps(o))
