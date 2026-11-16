"""Output formats for sharing: a few presets with the shapes social apps expect, and how the picture is
fitted into them. Files are H.264 video + AAC audio in .mp4 (plays everywhere), encoded with macOS's own
AVFoundation (see video_io). Covers are drawn on the full frame first, then the frame is cropped or
fitted, so framing can never uncover anyone.

Presets never upscale: a 720p clip exported as "Vertical 9:16" stays at its own resolution in that
shape (upscaling adds file size, not detail). Frame rates above 60 are reduced to ~30 for sharing.
"""
import cv2
import numpy as np

PRESETS = [
    {"id": "original", "label": "Original", "detail": "Same shape, size and frame rate as your clip", "aspect": None},
    {"id": "vertical", "label": "Vertical 9:16", "detail": "Reels, TikTok, YouTube Shorts, Stories", "aspect": (9, 16), "long": 1920},
    {"id": "portrait", "label": "Portrait 4:5", "detail": "Instagram and Facebook feed posts", "aspect": (4, 5), "long": 1350},
    {"id": "square", "label": "Square 1:1", "detail": "Square feed posts", "aspect": (1, 1), "long": 1080},
    {"id": "landscape", "label": "Landscape 16:9", "detail": "YouTube, X, Facebook, LinkedIn", "aspect": (16, 9), "long": 1920},
]
FITS = {"fill": "Crop to fill", "fit": "Whole frame, blurred background"}
BY_ID = {p["id"]: p for p in PRESETS}


def _even(x):
    return max(2, int(round(x / 2)) * 2)


def output_size(preset_id, src_w, src_h, fit="fill", photo=False):
    """(width, height) of the exported video (even, as H.264 needs), or photo: any size."""
    p = BY_ID.get(preset_id) or BY_ID["original"]
    ev = (lambda x: max(1, int(round(x)))) if photo else _even
    if not p["aspect"]:
        return ev(src_w), ev(src_h)
    a, b = p["aspect"]
    W, H = (p["long"], p["long"] * b / a) if a >= b else (p["long"] * a / b, p["long"])
    if fit == "fill":  # picture = the largest a:b crop of the source; don't enlarge it
        cw, ch = (src_h * a / b, src_h) if src_w / src_h > a / b else (src_w, src_w * b / a)
        k = min(1.0, cw / W)
    else:  # whole source inside the frame; don't enlarge it
        k = min(1.0, max(src_w / W, src_h / H))
    return ev(W * k), ev(H * k)


def _cover(img, W, H):
    h, w = img.shape[:2]
    s = max(W / w, H / h)
    nw, nh = max(W, round(w * s)), max(H, round(h * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    x, y = (nw - W) // 2, (nh - H) // 2
    return r[y:y + H, x:x + W]


def fit_frame(img, W, H, fit="fill"):
    """Fit a BGR frame into W x H: 'fill' = centre crop, 'fit' = whole frame over a blurred copy of itself."""
    h, w = img.shape[:2]
    if (w, h) == (W, H):
        return img
    if fit == "fill":
        return _cover(img, W, H)
    small = _cover(cv2.resize(img, (max(2, w // 8), max(2, h // 8)), interpolation=cv2.INTER_AREA),
                   max(2, W // 8), max(2, H // 8))
    bg = cv2.resize(cv2.GaussianBlur(small, (0, 0), 6), (W, H), interpolation=cv2.INTER_LINEAR)
    out = (bg.astype(np.float32) * 0.6).astype(np.uint8)
    s = min(W / w, H / h)
    nw, nh = max(2, round(w * s)), max(2, round(h * s))
    fg = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    x, y = (W - nw) // 2, (H - nh) // 2
    out[y:y + nh, x:x + nw] = fg
    return out


def settings(meta, preset_id="original", fit="fill", start=None, end=None, frames=None):
    """Everything the exporter needs, from the clip's probe() info and the user's choices."""
    fps = meta["fps"]
    n = frames or max(1, round(meta.get("video_duration", meta["duration"]) * fps))
    s = max(0, min(n - 1, round((start or 0) * fps)))
    e = n if end in (None, "", 0) else max(s + 1, min(n, round(float(end) * fps)))
    step = 1 if preset_id == "original" or fps <= 60 else max(1, round(fps / 30))
    W, H = output_size(preset_id, meta["width"], meta["height"], fit, photo=bool(meta.get("image")))
    out_fps = fps / step
    bitrate = int(min(40e6, max(2e6, 0.12 * W * H * out_fps)))  # ~0.12 bits per pixel: clean at 1080p30 (~7.5 Mb/s)
    return {"preset": preset_id, "fit": fit, "width": W, "height": H, "fps": out_fps, "step": step,
            "start": s, "end": e, "bitrate": bitrate, "seconds": (e - s) / fps}


def describe():
    """Presets + fits for the UI."""
    return {"presets": PRESETS, "fits": FITS}
