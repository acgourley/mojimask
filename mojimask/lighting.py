"""Scene-matched emoji lighting: each emoji is a matte cardboard cut-out standing where the face is.

Two layers:
  * AUTO estimate (computed once, cached): per frame the scene's colour cast and exposure, per track
    piece per frame the light on that face relative to its usual light, plus a light direction.
  * USER settings (edits["light"], the review tool's wheels/sliders) applied on top. The same maths
    runs in light_ui.js for the live preview -- keep `apply_settings` and `lightFor()` in sync.

Model, in LINEAR light (channels BGR in Python, RGB in JSON/JS):
    albedo = desat(albedo, saturation)               # the Saturation slider (0 = gray, 1 = the emoji's own)
    lit = albedo * (key * ramp(x, y) + ambient)
    lit = max(lit, albedo * floor)                  # legibility floor
    out = lerp(albedo, lit, mix)                     # mix 0 = flat emoji
    out = black + out * (1 - black)                  # FILM: the footage's black level (old film is never black)
  black = black_auto(t) * user black. Desaturating the albedo BEFORE the light keeps the scene's cast: on sepia
  footage gray emoji turn sepia.
      black_auto  = 1st-percentile luminance (linear), smoothed over time
      colour_auto = 90th-percentile pixel chroma / COLOUR_REF, capped at 1 (colour video ~0.6, B&W ~0): only
                    scales the colour part of the export's grain (none on black-and-white footage). Saturation
                    is a plain slider: on our clips this measure was only ever ~0 or ~1, so it added nothing.
  key_auto = cast(t) * white(t) * resid(t) ** local_k
      cast(t)  = the frame's white-patch colour cast, smoothed over time. ONE light colour per moment, shared
                 by every face: a face's own colour can't tell its skin from its light, and a colour averaged
                 over each track's lifetime made two covers on one head differ (pink vs lavender flicker).
      white(t) = white-patch luminance now: a white card is no brighter than the scene's brightest diffuse
                 surfaces (exposure anchor; carries strobes)
      resid(t) = this face's brightness now vs. its usual, divided by the scene's -- local spots (brightness
                 only). Comes from the noisy face probe, so it is smoothed and damped (power < 1). Dividing by
                 the piece's own median cancels skin tone: shading never encodes what the face looked like.
  key     = level * key_auto, or with a Light colour picked: its brightness (mean channel) in that colour
  ambient = amb * scene mean colour (always automatic)
  ramp    = 1 + ramp_strength * (gx * x + gy * y), from left/right and top/bottom face brightness
Export also adds grain matched to the frame and motion blur along the track's velocity.
"""
import colorsys
import hashlib
import json
import os

import cv2
import numpy as np

DEFAULTS = {"enabled": True, "level": 1.0, "mix": 1.0, "amb": 0.15, "ramp": 0.8, "floor": 0.12,
            "light_colour": None, "grain": True, "blur": True, "shutter": 1.0,
            "saturation": 1.0, "black": 1.0}
# saturation: the emoji's own colours, 0 = gray, 1 = as drawn (the default; the black-and-white sample starts at 0)
# black: multiplier on the footage's measured black level (1 = match it, 0 = emoji keep true black)
COLOUR_REF = 0.4  # 90th-percentile chroma at which footage counts as fully colourful
# shutter = motion blur length as a fraction of one frame's movement (1.0 = shutter open the whole frame,
# typical for phone video in a dark room; 0 = off). "blur": False (older setting) also turns it off.
NEAR_FRAMES = 15  # a frame without its own estimate borrows the piece's nearest one within this many (sent to light_ui.js)
# light_colour: None = the estimated light colour (per face, per frame), or [hue_deg, radius 0..1] from the
# Light colour wheel: HSV (hue, radius * WHEEL_SAT, 1), so the wheel's centre is white light.
# mix ("Strength"): 1 = fully lit, 0 = the flat emoji.
WHEEL_SAT = 0.8
LOCAL_K, LOCAL_A = 0.6, 0.25
OVERLAP = 0.5  # covers whose centres are within this x their size of each other share one light estimate


def settings(edits):
    return {**DEFAULTS, **((edits or {}).get("light") or {})}


# ------------------------------------------------------------------ colour helpers
def to_lin(x):  # sRGB 0..1 -> linear
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def lum(x):  # BGR linear
    return 0.0722 * x[..., 0] + 0.7152 * x[..., 1] + 0.2126 * x[..., 2]


def chroma(c):  # normalise so mean channel = 1
    return np.asarray(c, float) / max(float(np.mean(c)), 1e-6)


def wheel_bgr(wheel):
    """Light colour wheel [hue_deg, radius] -> light colour normalised to mean 1, BGR (light_ui.js wheelRGB)."""
    hue, amt = wheel
    r, g, b = colorsys.hsv_to_rgb((float(hue) % 360) / 360, min(max(float(amt), 0), 1) * WHEEL_SAT, 1)
    return chroma([b, g, r])


# ------------------------------------------------------------------ auto estimate
def scene_stats(f_lin):
    """White-patch cast (top-5% luminance, unclipped), mean colour, white-patch luminance."""
    L = lum(f_lin)
    ok = f_lin.max(-1) < 0.95
    if ok.sum() < 50:
        ok = np.ones_like(L, bool)
    sel = ok & (L >= np.percentile(L[ok], 95))
    return chroma(f_lin[sel].mean(0)), f_lin.reshape(-1, 3).mean(0), float(L[sel].mean())


def film_stats(f_srgb, f_lin):
    """(colourfulness 0..1, black level linear) of one frame: what the camera/film did, not the light."""
    ch = f_srgb.max(-1) - f_srgb.min(-1)
    return min(float(np.percentile(ch, 90)) / COLOUR_REF, 1.0), float(np.percentile(lum(f_lin), 1))


def desat(x, k):
    """Linear BGR towards its own luminance; k = 1 unchanged, 0 gray, >1 more saturated."""
    L = (0.0722 * x[..., 0] + 0.7152 * x[..., 1] + 0.2126 * x[..., 2])[..., None]
    return np.maximum(L + k * (x - L), 0)


_DESAT = {}


def desat_sprite(sprite_rgba, k):
    """An RGBA uint8 sprite (RGB order) with its colours at saturation k, for pasting without lighting.
    Same maths as lit_sprite's first step; cached per sprite and k."""
    key = (id(sprite_rgba), round(float(k), 3))
    if key not in _DESAT:
        if len(_DESAT) > 256:
            _DESAT.clear()
        rgb = to_lin(sprite_rgba[..., 2::-1] / 255.0)  # BGR linear
        out = to_srgb(desat(rgb, k))[..., ::-1]
        _DESAT[key] = (sprite_rgba, np.concatenate([(out * 255 + 0.5).astype(np.uint8), sprite_rgba[..., 3:]], -1))
    return _DESAT[key][1]


def face_probe(f_lin, c):
    """Mean colour of the inner face, and a left/right, top/bottom brightness gradient."""
    H, W = f_lin.shape[:2]
    r = c[2] * 0.28  # inner part of the head box: mostly face, little background
    x1, y1, x2, y2 = int(max(c[0] - r, 0)), int(max(c[1] - r, 0)), int(min(c[0] + r, W)), int(min(c[1] + r, H))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    p = f_lin[y1:y2, x1:x2]
    L = lum(p)
    mx, my = L.shape[1] // 2, L.shape[0] // 2
    d = max(L.mean() * 2, 1e-6)
    g = np.clip([(L[:, mx:].mean() - L[:, :mx].mean()) / d, (L[my:].mean() - L[:my].mean()) / d], -0.5, 0.5)
    return p.reshape(-1, 3).mean(0), g


def smooth_both_ways(rows, a):
    """Rows over consecutive frames, EMA forwards then backwards: smooth without lagging behind."""
    f = smooth_series(dict(enumerate(rows)), a)
    b = smooth_series({-t: v for t, v in f.items()}, a)
    return np.array([b[-t] for t in range(len(rows))])


def smooth_series(d, a=0.5):
    """EMA over consecutive frames (restarts after gaps)."""
    out, prev, pt = {}, None, None
    for t in sorted(d):
        v = np.asarray(d[t], float)
        prev = v if prev is None or t - pt > 2 else a * v + (1 - a) * prev
        out[t], pt = prev, t
    return out


def _frame(frames_dir, t):
    """Frame t of frames_dir as BGR float 0..1 (sRGB)."""
    p = f"{frames_dir}/{t:05d}.jpg"
    f = cv2.imread(p)
    if f is None:
        raise RuntimeError(f"lighting: can't read frame {p}")
    return f / 255.0


class Lighting:
    """snap: {piece: {t(int or str): [cx, cy, s]}} automatic track pieces. frames_dir: NNNNN.jpg frames."""

    def __init__(self, snap, frames_dir, cache_dir, progress=None):
        self.snap = {str(p): {int(t): [float(v) for v in c] for t, c in b.items()} for p, b in snap.items()}
        n = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        self.colour, self.black = self._film_estimate(frames_dir, cache_dir, n)
        sm = lambda v: np.array([x for _, x in sorted(smooth_series(dict(enumerate(v)), a=0.15).items())])
        self.colour, self.black = sm(self.colour), sm(self.black)
        self.cast, self.mean, self.white, probe = self._scene_estimate(frames_dir, cache_dir, n, progress)
        self.n = len(self.white)
        self.cast_s = np.array([chroma(c) for c in smooth_both_ways(list(self.cast), 0.3)])  # per-frame light colour
        self.key, self.grad = {}, {}
        for p, pr in probe.items():
            if len(pr) < 3:
                continue
            med = np.median([v[0] for v in pr.values()], 0)
            wmed = float(np.median([self.white[t] for t in pr]))
            scene = {t: self.white[t] / wmed for t in pr}
            ratio = {t: float(np.mean(np.clip(v[0] / np.maximum(med, 1e-4), 0.2, 3.0))) for t, v in pr.items()}
            resid = smooth_series({t: np.clip(ratio[t] / scene[t], 0.33, 3.0) for t in pr}, a=LOCAL_A)
            self.key[p] = {t: self.cast_s[t] * self.white[t] * float(resid[t]) ** LOCAL_K for t in pr}
            self.grad[p] = smooth_series({t: v[1] for t, v in pr.items()}, a=0.25)
        self._share_overlaps()

    @staticmethod
    def _film_estimate(frames_dir, cache_dir, n):
        """Per frame (colourfulness, black level), cached: independent of the tracks, but not of the frames (the
        first frame's size and time are in the cache's name, so frames extracted again are measured again)."""
        st = os.stat(f"{frames_dir}/00000.jpg") if n else None
        tag = f"{n}_{st.st_size}_{int(st.st_mtime)}" if st else f"{n}"
        film = f"{cache_dir}/film_est_{tag}.npz"
        if os.path.exists(film):
            z = np.load(film)
            return z["colour"], z["black"]
        colour, black = np.ones(n), np.zeros(n)
        for t in range(n):
            fs = _frame(frames_dir, t)
            colour[t], black[t] = film_stats(fs, to_lin(fs))
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(film, colour=colour, black=black)
        return colour, black

    def _scene_estimate(self, frames_dir, cache_dir, n, progress=None):
        """Per frame (cast, mean colour, white-patch luminance) and per piece {t: face_probe}, cached per set of
        track boxes."""
        key = hashlib.md5(json.dumps({p: sorted(b.items()) for p, b in sorted(self.snap.items())},
                                     default=float).encode()).hexdigest()[:12]
        cache = f"{cache_dir}/light_est_{key}.npz"
        if os.path.exists(cache):
            z = np.load(cache, allow_pickle=True)
            return z["cast"], z["mean"], z["white"], z["probe"].item()
        cast, mean, white = np.ones((n, 3)), np.zeros((n, 3)), np.ones(n)
        by_t = {}
        for p, b in self.snap.items():
            for t, c in b.items():
                by_t.setdefault(t, []).append((p, c))
        probe = {p: {} for p in self.snap}
        for t in range(n):
            fl = to_lin(_frame(frames_dir, t))
            cast[t], mean[t], white[t] = scene_stats(fl)
            for p, c in by_t.get(t, []):
                pr = face_probe(fl, c)
                if pr is not None:
                    probe[p][t] = pr
            if progress and t % 50 == 0:
                progress(t / n)
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache, cast=cast, mean=mean, white=white, probe=np.array(probe, dtype=object))
        return cast, mean, white, probe

    def _share_overlaps(self):
        """Covers sitting on the same spot at the same moment (two track pieces on one head) are under the same
        light: give each group their average, or whichever is drawn on top flips the emoji's shade frame to frame."""
        by_t = {}
        for p, keys in self.key.items():
            for t in keys:
                c = self.snap[p].get(t)
                if c is not None:
                    by_t.setdefault(t, []).append((p, c))
        for t, items in by_t.items():
            if len(items) < 2:
                continue
            group = {p: p for p, _ in items}
            find = lambda p: p if group[p] == p else find(group[p])
            for i, (p, a) in enumerate(items):
                for q, b in items[i + 1:]:
                    if abs(a[0] - b[0]) < OVERLAP * max(a[2], b[2]) and abs(a[1] - b[1]) < OVERLAP * max(a[2], b[2]):
                        group[find(q)] = find(p)
            groups = {}
            for p, _ in items:
                groups.setdefault(find(p), []).append(p)
            for ps in groups.values():
                if len(ps) > 1:
                    k = np.mean([self.key[p][t] for p in ps], 0)
                    g = np.mean([self.grad[p][t] for p in ps], 0)
                    for p in ps:
                        self.key[p][t], self.grad[p][t] = k, g

    def auto(self, piece, t):
        """(key BGR, grad xy, ambient chroma BGR) before user settings."""
        t = min(max(int(t), 0), self.n - 1)
        keys = self.key.get(piece, {})
        k, g = keys.get(t), self.grad.get(piece, {}).get(t)
        if k is None and keys:
            # a frame the probe didn't measure (failed probe, or a flow-carried cover past the track's boxes):
            # the piece's nearest measured frame if it is close, so the emoji's colour doesn't jump
            u = min(keys, key=lambda u: abs(u - t))
            if abs(u - t) <= NEAR_FRAMES:
                k, g = keys[u], self.grad[piece].get(u)
        if k is None:  # manual piece / no usable probe: scene cast and exposure only
            k = self.cast_s[t] * self.white[t]
        return k, (g if g is not None else np.zeros(2)), chroma(self.mean[t])

    def film(self, t):
        """(colourfulness, black level) of the footage at frame t, before user settings."""
        t = min(max(int(t), 0), self.n - 1)
        return float(self.colour[t]), float(self.black[t])

    def velocity(self, piece, t):
        b = self.snap.get(piece)
        if not b or t not in b:
            return 0.0, 0.0
        p, q = b.get(t - 1, b[t]), b.get(t + 1, b[t])
        return (q[0] - p[0]) / 2, (q[1] - p[1]) / 2

    def to_json(self):
        """Compact payload for the review tool (RGB order): per-frame fallback key + ambient chroma,
        per-piece per-frame [kr, kg, kb, gx, gy], and NEAR_FRAMES (light_ui.js borrows estimates as far)."""
        r3 = lambda v: [round(float(x), 4) for x in v[::-1]]
        return {"ready": True, "n": self.n, "defaults": DEFAULTS, "wheel_sat": WHEEL_SAT, "near_frames": NEAR_FRAMES,
                "scene_key": [r3(self.cast_s[t] * self.white[t]) for t in range(self.n)],
                "scene_amb": [r3(chroma(self.mean[t])) for t in range(self.n)],
                "film": [[round(float(self.colour[t]), 3), round(float(self.black[t]), 5)] for t in range(self.n)],
                "pieces": {p: {str(t): r3(k[t]) + [round(float(g), 3) for g in self.grad[p][t]] for t in k}
                           for p, k in self.key.items()}}


def apply_settings(key, grad, amb_chroma, st):
    """Auto estimate + user settings -> (key, ambient, ramp gradient). Mirrors light_ui.js lightFor()."""
    key = st["level"] * np.asarray(key, float)
    if st.get("light_colour"):
        key = float(np.mean(key)) * wheel_bgr(st["light_colour"])
    return key, st["amb"] * np.asarray(amb_chroma), np.asarray(grad) * st["ramp"]


def film_settings(colour_auto, black_auto, st):
    """User saturation + the footage's measured black level x the user's multiplier -> (saturation, black).
    Mirrors light_ui.js filmFor(). colour_auto isn't used here (only for grain)."""
    return min(max(float(st["saturation"]), 0.0), 1.0), min(max(black_auto * float(st["black"]), 0.0), 0.5)


# ------------------------------------------------------------------ compositing
def lit_sprite(sprite_rgba, key, amb, grad, mix, floor, saturation=1.0, black=0.0):
    """sprite RGBA uint8 (RGB order) -> BGRA float 0..1 (straight alpha): its colours at the user's saturation,
    lit, then given the footage's black level."""
    alb = to_lin(sprite_rgba[..., 2::-1] / 255.0)
    if saturation != 1.0:
        alb = desat(alb, saturation)
    a = sprite_rgba[..., 3:4] / 255.0
    s = sprite_rgba.shape[0]
    yy, xx = np.mgrid[0:s, 0:s] / max(s - 1, 1) * 2 - 1  # -1..1 across the card
    ramp = np.clip(1 + grad[0] * xx + grad[1] * yy, 0.2, 2.0)[..., None]
    lit = np.maximum(alb * (key[None, None, :] * ramp + amb[None, None, :]), alb * floor)
    out = (1 - mix) * alb + mix * lit
    return np.concatenate([to_srgb(black + out * (1 - black)), a], -1).astype(np.float32)


def motion_kernel(vx, vy, shutter=1.0):
    L = float(np.hypot(vx, vy) * shutter)
    if L < 1.0:
        return None
    k = int(np.ceil(L)) | 1
    ker = np.zeros((k, k), np.float32)
    c, d = k // 2, max(np.hypot(vx, vy), 1e-6)
    for s in np.linspace(-L / 2, L / 2, k * 3):
        ker[int(round(c + s * vy / d)), int(round(c + s * vx / d))] = 1
    return ker / ker.sum()


def frame_grain(f):
    """Robust noise sigma (8-bit units) of a BGR uint8 frame: the median deviation from a slightly blurred copy, read
    on 16 tiles spread over the frame (a quarter of it; within ~10% of the whole frame's, 3-4x faster)."""
    H, W = f.shape[:2]
    th, tw = max(H // 8, 16), max(W // 8, 16)
    ds = []
    for i in range(4):
        for j in range(4):
            y, x = max((2 * i + 1) * H // 8 - th // 2, 0), max((2 * j + 1) * W // 8 - tw // 2, 0)
            g = f[y:y + th, x:x + tw]
            if H > 1000:
                g = cv2.resize(g, None, fx=0.5, fy=0.5)
            g = g.astype(np.float32)
            ds.append(np.abs(g - cv2.GaussianBlur(g, (0, 0), 1.0))[2:-2, 2:-2].ravel())
    return float(np.median(np.concatenate(ds)) * 1.4826)


def paste_lit(f, spr, c, k, rng=None, grain=0.0, vel=None, shutter=1.0, colourfulness=1.0):
    """Composite a lit BGRA float sprite onto uint8 BGR frame f in place. colourfulness: the footage's (0..1),
    how much of the grain is colour."""
    s = int(round(c[2] * k))
    if s < 4:
        return
    sp = cv2.resize(spr, (s, s), interpolation=cv2.INTER_AREA)
    pm = np.concatenate([sp[..., :3] * sp[..., 3:], sp[..., 3:]], -1)  # premultiply before blurring
    ker = motion_kernel(*vel, shutter) if vel is not None and shutter > 0 else None
    if ker is not None:
        pad = ker.shape[0]
        pm = cv2.filter2D(cv2.copyMakeBorder(pm, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0), -1, ker)
        s += 2 * pad
    pm = cv2.GaussianBlur(pm, (0, 0), 0.6)  # sprites are sharper than phone video
    if rng is not None and grain > 0:
        # luminance grain, plus colour grain only as far as the footage has colour (none on black and white)
        n = rng.normal(0, grain / 255.0, pm.shape[:2] + (1,)) + min(colourfulness, 1.0) * rng.normal(0, grain / 510.0, pm.shape[:2] + (3,))
        pm[..., :3] += n.astype(np.float32) * pm[..., 3:]
    x0, y0 = int(round(c[0] - s / 2)), int(round(c[1] - s / 2))
    H, W = f.shape[:2]
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + s, W), min(y0 + s, H)
    if xa >= xb or ya >= yb:
        return
    pm = pm[ya - y0:yb - y0, xa - x0:xb - x0]
    bg = f[ya:yb, xa:xb].astype(np.float32) / 255.0
    f[ya:yb, xa:xb] = (np.clip(pm[..., :3] + (1 - pm[..., 3:]) * bg, 0, 1) * 255 + 0.5).astype(np.uint8)


class FrameLighter:
    """Per-render helper used by track.draw_frame: paste(f, sprite, c, scale, piece, t).
    per_frame: the resolved boxes being rendered ([(tid, c, kind, piece), ...] per frame); motion blur
    velocity comes from these (covers manual pieces too, matches the preview). Without it, auto tracks only."""

    def __init__(self, lighting, st, per_frame=None):
        self.L, self.st = lighting, {**DEFAULTS, **st}
        self._per_t = {}
        self.boxes = None
        if per_frame is not None:
            self.boxes = [{it[3] if len(it) > 3 else it[0]: it[1] for it in f} for f in per_frame]

    def velocity(self, piece, t):
        if self.boxes is None:
            return self.L.velocity(piece, t)
        cur = self.boxes[t].get(piece) if 0 <= t < len(self.boxes) else None
        if cur is None:
            return 0.0, 0.0
        p = self.boxes[t - 1].get(piece, cur) if t > 0 else cur
        q = self.boxes[t + 1].get(piece, cur) if t + 1 < len(self.boxes) else cur
        return (q[0] - p[0]) / 2, (q[1] - p[1]) / 2

    def params(self, piece, t, f):
        """Everything needed to light cover `piece` on frame t (f: that frame, for its grain), as a dict: key, amb,
        grad, mix, floor, black, saturation, colourfulness (the footage's), grain, rng (None without grain),
        shutter (0 = no motion blur) and vel (the cover's movement per frame, None without blur). paste() and
        gpu_draw both use it."""
        st = self.st
        # once per frame, kept per frame number: the editor's server draws several frames at once on threads,
        # so a single "current frame" slot would hand one frame's grain/noise to another
        if t not in self._per_t:
            if len(self._per_t) > 64:
                self._per_t.clear()
            self._per_t[t] = (frame_grain(f) if st["grain"] else 0.0, np.random.default_rng(t))
        grain, rng = self._per_t[t]
        key, amb, grad = apply_settings(*self.L.auto(piece, t), st)
        film = self.L.film(t)
        saturation, black = film_settings(*film, st)
        shutter = float(st["shutter"]) if st["blur"] else 0.0
        return {"key": key, "amb": amb, "grad": grad, "mix": st["mix"], "floor": st["floor"], "black": black,
                "saturation": saturation, "colourfulness": film[0], "grain": grain if st["grain"] else 0.0,
                "rng": rng if st["grain"] else None, "shutter": shutter,
                "vel": self.velocity(piece, t) if shutter > 0 else None}

    def paste(self, f, sprite, c, scale, piece, t):
        p = self.params(piece, t, f)
        s = int(round(c[2] * scale))
        if s < 4:
            return
        if sprite.shape[0] > s:  # light at the size it is drawn (the lighting is resolution-free; full-size
            sprite = cv2.resize(sprite, (s, s), interpolation=cv2.INTER_AREA)  # posed photos cost ~9x otherwise)
        spr = lit_sprite(sprite, p["key"], p["amb"], p["grad"], p["mix"], p["floor"], p["saturation"], p["black"])
        paste_lit(f, spr, c, scale, p["rng"], p["grain"], p["vel"], p["shutter"], p["colourfulness"])  # grain is the footage's
