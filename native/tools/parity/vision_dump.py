"""Write cv2 inputs and outputs for mm-vision's parity check (native/tools/parity/vision_check.py runs this).

Every case: the function, its arguments, input arrays and cv2's output, as .npy files plus one manifest.json in
OUT (default ROOT/out/native/vision_parity). Inputs are jazz frames (public domain, out/poc/data/jazz/in) and
seeded random images of odd sizes, covering every argument combination the engine uses (grep cv2. in mojimask/)
plus edge cases.

    PYTHONPATH=ROOT .venv/bin/python native/tools/parity/vision_dump.py [OUT]
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out", "native", "vision_parity")
JAZZ = os.path.join(ROOT, "out", "poc", "data", "jazz", "in")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261117)
cases = []
_n = [0]


def arr(a):
    _n[0] += 1
    name = f"a{_n[0]:05d}.npy"
    np.save(os.path.join(OUT, name), np.ascontiguousarray(a))
    return name


def case(name, fn, args, inputs, out):
    if isinstance(out, (list, tuple)) or np.isscalar(out):
        out = np.asarray(out, np.float64)
    cases.append({"name": name, "fn": fn, "args": args, "inputs": [arr(a) for a in inputs], "output": arr(out)})


def jazz(i):
    return cv2.imread(os.path.join(JAZZ, f"{i:05d}.png"))


def rand_img(h, w, cn, dtype=np.uint8, smooth=False):
    if dtype == np.uint8:
        a = rng.integers(0, 256, (h, w, cn), dtype=np.uint8)
        if smooth:  # natural-looking: blocky gradients so interpolation rounding is exercised mid-range
            a = cv2.resize(rng.integers(0, 256, (max(h // 7, 2), max(w // 7, 2), cn), dtype=np.uint8).reshape(max(h // 7, 2), max(w // 7, 2), cn), (w, h), interpolation=cv2.INTER_CUBIC).reshape(h, w, cn)
    elif dtype == np.float64:
        a = rng.normal(0, 1, (h, w, cn)) * 40 + 100
    else:
        a = rng.normal(0, 1, (h, w, cn)).astype(np.float32) * 40 + 100
    return a[..., 0] if cn == 1 else a


J = [jazz(i) for i in (0, 37, 101)]
JG = [cv2.cvtColor(j, cv2.COLOR_BGR2GRAY) for j in J]
JF = [j.astype(np.float32) / 255 for j in J]

# ---------------- cvtColor ----------------
for i, (h, w) in enumerate([(1, 1), (3, 7), (5, 15), (9, 16), (11, 17), (31, 33), (64, 129)]):
    a = rand_img(h, w, 3)
    case(f"cvt_gray_rand{h}x{w}", "cvt_bgr2gray", {}, [a], cv2.cvtColor(a, cv2.COLOR_BGR2GRAY))
    case(f"cvt_rgb_rand{h}x{w}", "cvt_swap_rb", {}, [a], cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
    f = rand_img(h, w, 3, np.float32)
    case(f"cvt_rgb_f32_{h}x{w}", "cvt_swap_rb", {}, [f], cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
for k, j in enumerate(J):
    case(f"cvt_gray_jazz{k}", "cvt_bgr2gray", {}, [j], cv2.cvtColor(j, cv2.COLOR_BGR2GRAY))
    case(f"cvt_rgb_jazz{k}", "cvt_swap_rb", {}, [j], cv2.cvtColor(j, cv2.COLOR_BGR2RGB))
a = np.full((4, 20, 3), 255, np.uint8)
case("cvt_gray_white", "cvt_bgr2gray", {}, [a], cv2.cvtColor(a, cv2.COLOR_BGR2GRAY))

# ---------------- resize ----------------
INTERS = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA}


def rs_dsize(tag, a, w, h, inter):
    case(f"resize_{tag}_{inter}_{a.shape[1]}x{a.shape[0]}to{w}x{h}", "resize",
         {"dw": w, "dh": h, "fx": 0.0, "fy": 0.0, "interp": inter}, [a],
         cv2.resize(a, (w, h), interpolation=INTERS[inter]))


def rs_f(tag, a, fx, fy, inter):
    case(f"resize_{tag}_{inter}_{a.shape[1]}x{a.shape[0]}_f{fx:.4g}x{fy:.4g}", "resize",
         {"dw": 0, "dh": 0, "fx": fx, "fy": fy, "interp": inter}, [a],
         cv2.resize(a, None, fx=fx, fy=fy, interpolation=INTERS[inter]))


for inter in INTERS:
    for k, j in enumerate(J[:2]):
        for (w, h) in [(1280, 948), (640, 474), (321, 238), (214, 159), (160, 120), (112, 112), (900, 300), (643 * 2, 476 * 2)]:
            rs_dsize(f"jazz{k}", j, w, h, inter)
        rs_dsize(f"jazzg{k}", JG[k], 321, 238, inter)
        rs_dsize(f"jazzg{k}", JG[k], 1000, 700, inter)
        rs_dsize(f"jazzf{k}", JF[k], 321, 238, inter)
        rs_dsize(f"jazzf{k}", JF[k], 1000, 700, inter)
        for fx in (0.5, 0.25, 1 / 3, 0.37, 2.0, 1.7):
            rs_f(f"jazz{k}", j, fx, fx, inter)
    for cn in (1, 2, 3, 4):
        for dt, dn in ((np.uint8, "u8"), (np.float32, "f32")):
            if cn == 2 and dt == np.float32:
                continue
            for (h, w) in [(37, 53), (64, 96), (17, 9), (128, 128), (5, 3)]:
                a = rand_img(h, w, cn, dt, smooth=(dt == np.uint8 and h > 20))
                for (dw, dh) in [(w // 2, h // 2), (w * 2, h * 2), (w // 3 or 1, h // 3 or 1), (int(w * 0.61) or 1, int(h * 0.77) or 1),
                                 (int(w * 1.37), int(h * 1.61)), (w + 1, h - 1 if h > 1 else 1), (w // 4 or 1, h // 4 or 1), (8, 8), (19, 23)]:
                    rs_dsize(f"r{dn}c{cn}", a, dw, dh, inter)
                for fx, fy in ((0.5, 0.5), (0.25, 0.25), (2.0, 2.0), (0.5, 0.25), (1 / 3, 1 / 3), (0.7, 0.45)):
                    if int(w * fx) > 0 and int(h * fy) > 0:
                        rs_f(f"r{dn}c{cn}", a, fx, fy, inter)
# the engine's own calls
uv = rng.integers(0, 256, (119, 161, 2), dtype=np.uint8)
rs_dsize("nv12uv", uv, 322, 238, "nearest")
sprite = rng.random((512, 512, 4)).astype(np.float32)
for s in (37, 128, 256, 300, 511):
    rs_dsize("spritef", sprite, s, s, "area")
sp8 = rng.integers(0, 256, (512, 512, 4), dtype=np.uint8)
for s in (37, 128, 256, 300):
    rs_dsize("sprite8", sp8, s, s, "area")
dep = rng.random((476, 643)).astype(np.float32) * 3 - 1
rs_dsize("depthf", dep, 256, 189, "area")
rs_dsize("depthf", dep, 128, 95, "area")

# float64 (depth.py: decoded bytes times the frame's float64 fit, area-shrunk; the occluder map eroded)
for cn in (1, 3, 4):
    for (h, w) in [(37, 53), (64, 96), (17, 9), (128, 128), (5, 3), (476, 643)]:
        a = rand_img(h, w, cn, np.float64)
        for (dw, dh) in [(w // 2, h // 2), (w // 3 or 1, h // 3 or 1), (int(w * 0.61) or 1, int(h * 0.77) or 1), (w // 4 or 1, h // 4 or 1),
                         (w - 1, h), (max(w // 5, 1), max(h // 2, 1)), (1, 1), (8, 8)]:
            if dw <= w and dh <= h:
                rs_dsize(f"rf64c{cn}", a, dw, dh, "area")
                rs_dsize(f"rf64c{cn}", a, dw, dh, "nearest")
        for fx, fy in ((0.5, 0.5), (0.25, 0.25), (0.5, 0.25), (1 / 3, 1 / 3), (0.7, 0.45), (0.2, 0.2)):
            if int(w * fx) > 0 and int(h * fy) > 0:
                rs_f(f"rf64c{cn}", a, fx, fy, "area")
        if h >= 2 and w >= 2:
            rs_f(f"rf64c{cn}", a, 0.5, 0.5, "linear")  # LINEAR at exactly 2x down is AREA
for k, j in enumerate(JG[:2]):  # depth.py's grid: luma bytes * a + b in float64, shrunk to CACHE_SIDE
    d64 = j.astype(np.float32).astype(np.float64) * rng.uniform(0.002, 0.02) + rng.uniform(-1, 1)
    for (w, h) in [(256, 189), (128, 95), (321, 238), (160, 119)]:
        rs_dsize(f"depthf64_{k}", d64, w, h, "area")

# ---------------- warpAffine ----------------
BORDERS = {"constant": cv2.BORDER_CONSTANT, "replicate": cv2.BORDER_REPLICATE}


def warp(tag, a, M, w, h, inverse, border, bval=0.0):
    flags = cv2.INTER_LINEAR | (cv2.WARP_INVERSE_MAP if inverse else 0)
    out = cv2.warpAffine(a, M, (w, h), flags=flags, borderMode=BORDERS[border], borderValue=(bval, bval, bval, bval))
    case(f"warp_{tag}_{w}x{h}_{'inv' if inverse else 'fwd'}_{border}{bval:g}", "warp_affine",
         {"m": [float(v) for v in np.asarray(M, np.float64).ravel()], "w": w, "h": h, "inverse": inverse, "border": border, "bval": bval},
         [a], out)


def sim(scale, ang, tx, ty):
    c, s = np.cos(ang) * scale, np.sin(ang) * scale
    return np.array([[c, -s, tx], [s, c, ty]])


for k, (j, g, f) in enumerate(zip(J, JG, JF)):
    for t in range(4):
        M = sim(rng.uniform(0.3, 2.5), rng.uniform(-0.6, 0.6), rng.uniform(-200, 100), rng.uniform(-150, 80))
        for inv in (False, True):
            warp(f"jazz{k}t{t}", j, M, 112, 112, inv, "constant")
            warp(f"jazzg{k}t{t}", g, M.astype(np.float32), 61, 59, inv, "constant")
            warp(f"jazzf{k}t{t}", g.astype(np.float32) / 255, M, 133, 97, inv, "replicate")
            warp(f"jazz{k}t{t}", j, M, 203, 157, inv, "replicate")
            warp(f"jazzf3{k}t{t}", f, M, 45, 37, inv, "constant", 0.5)
    # depth.py's inverse-map crop (float32 matrix, replicate)
    near = rng.random((189, 256)).astype(np.float32)
    H, W = j.shape[:2]
    kx, ky = 256 / W, 189 / H
    for (x1, y1, x2, y2) in [(10, 20, 91, 117), (0, 0, 643, 476), (500, 300, 643, 476), (200, 100, 237, 133)]:
        M = np.float32([[kx, 0, (x1 + 0.5) * kx - 0.5], [0, ky, (y1 + 0.5) * ky - 0.5]])
        warp(f"depthcrop{k}", near, M, x2 - x1, y2 - y1, True, "replicate")
for cn in (1, 3, 4):
    for dt in (np.uint8, np.float32):
        for (h, w) in [(1, 1), (2, 3), (37, 53), (64, 64)]:
            a = rand_img(h, w, cn, dt)
            for t in range(3):
                M = np.array([[rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-30, 30)], [rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-30, 30)]])
                for inv in (False, True):
                    for b in ("constant", "replicate"):
                        warp(f"r{np.dtype(dt).name}c{cn}_{h}x{w}t{t}", a, M, 29, 23, inv, b, 7.0 if b == "constant" else 0.0)
        # exact integer and half-pixel shifts, identity, pure scale
        a = rand_img(40, 50, cn, dt)
        for M in ([[1, 0, 0], [0, 1, 0]], [[1, 0, 0.5], [0, 1, -0.5]], [[1, 0, 3], [0, 1, -2]], [[2, 0, 0], [0, 2, 0]], [[0.5, 0, 0.25], [0, 0.5, 0.25]], [[0, 1, 0], [-1, 0, 49]]):
            warp(f"exact{np.dtype(dt).name}c{cn}", a, np.array(M, np.float64), 47, 41, True, "constant")
            warp(f"exact{np.dtype(dt).name}c{cn}", a, np.array(M, np.float64), 47, 41, False, "replicate")

# ---------------- GaussianBlur ----------------


def gauss(tag, a, kw, kh, sx, sy=0.0):
    out = cv2.GaussianBlur(a, (kw, kh), sx, sigmaY=sy)
    case(f"gauss_{tag}_{a.dtype.name}_{a.shape[1]}x{a.shape[0]}_k{kw}x{kh}_s{sx:g}_{sy:g}", "gaussian_blur",
         {"kw": kw, "kh": kh, "sx": sx, "sy": sy}, [a], out)


for k in range(2):
    for (kw, kh, sx, sy) in [(3, 3, 0, 0), (5, 5, 0, 0), (7, 7, 0, 0), (9, 9, 0, 0), (11, 11, 0, 0), (21, 21, 0, 0), (0, 0, 6, 0), (0, 0, 1.0, 0), (0, 0, 0.6, 0), (5, 3, 0, 0), (0, 0, 2.3, 1.1), (13, 13, 3.3, 0)]:
        gauss(f"jazz{k}", J[k], kw, kh, sx, sy)
        gauss(f"jazzg{k}", JG[k], kw, kh, sx, sy)
        gauss(f"jazzf{k}", JF[k], kw, kh, sx, sy)
        gauss(f"jazzgf{k}", JG[k].astype(np.float32) / 255, kw, kh, sx, sy)
for cn in (1, 3, 4):
    for dt in (np.uint8, np.float32):
        for (h, w) in [(1, 1), (1, 9), (7, 1), (2, 2), (5, 6), (13, 17), (40, 33)]:
            a = rand_img(h, w, cn, dt)
            for (kw, kh, sx, sy) in [(3, 3, 0, 0), (5, 5, 0, 0), (0, 0, 1.0, 0), (0, 0, 0.6, 0), (0, 0, 6.0, 0), (9, 3, 0, 0), (15, 15, 0, 0)]:
                gauss(f"r{cn}", a, kw, kh, sx, sy)
# float rows/columns of 3 and 5 taps with sigma > 0 (non-dyadic weights), odd and even row lengths
for (h, w, cn) in [(40, 33, 1), (21, 34, 1), (9, 11, 3), (12, 7, 4)]:
    a = rand_img(h, w, cn, np.float32)
    for (kw, kh, sx, sy) in [(3, 3, 0.8, 0), (5, 5, 1.3, 0), (5, 1, 0, 0), (1, 5, 0, 0), (3, 5, 0.7, 1.1), (7, 3, 0, 0.9)]:
        gauss(f"taps{cn}", a, kw, kh, sx, sy)
# lighting's text shadow (float alpha)
a = (rng.random((61, 211)) > 0.7).astype(np.float32)
for s in (1.0, 1.7, 4.2):
    gauss("shadow", a, 0, 0, s)

# ---------------- erode ----------------
for dt in (np.uint8, np.float32):
    for cn in (1, 3):
        for (h, w) in [(1, 1), (3, 2), (17, 23), (64, 97)]:
            a = rand_img(h, w, cn, dt)
            for ks in (1, 3, 5, 7, 11):
                case(f"erode_{np.dtype(dt).name}c{cn}_{h}x{w}_k{ks}", "erode", {"kw": ks, "kh": ks}, [a],
                     cv2.erode(a, np.ones((ks, ks), np.uint8)))
            case(f"erode_{np.dtype(dt).name}c{cn}_{h}x{w}_k5x3", "erode", {"kw": 5, "kh": 3}, [a], cv2.erode(a, np.ones((3, 5), np.uint8)))
for cn in (1, 3):
    for (h, w) in [(1, 1), (3, 2), (17, 23), (64, 97)]:
        a = rand_img(h, w, cn, np.float64)
        for ks in (1, 2, 3, 5, 7, 11):
            case(f"erode_float64c{cn}_{h}x{w}_k{ks}", "erode", {"kw": ks, "kh": ks}, [a], cv2.erode(a, np.ones((ks, ks), np.uint8)))
        case(f"erode_float64c{cn}_{h}x{w}_k5x3", "erode", {"kw": 5, "kh": 3}, [a], cv2.erode(a, np.ones((3, 5), np.uint8)))
m64 = (JG[1].astype(np.float64) - 100) / 30
for g in (1, 2, 3, 4):
    case(f"erode_depthmap_f64_g{g}", "erode", {"kw": 2 * g + 1, "kh": 2 * g + 1}, [m64], cv2.erode(m64, np.ones((2 * g + 1, 2 * g + 1), np.uint8)))
m = (JG[0].astype(np.float32) - 100) / 30
for g in (1, 2, 4):
    case(f"erode_depthmap_g{g}", "erode", {"kw": 2 * g + 1, "kh": 2 * g + 1}, [m], cv2.erode(m, np.ones((2 * g + 1, 2 * g + 1), np.uint8)))

# ---------------- copyMakeBorder ----------------
for dt in (np.uint8, np.float32):
    for cn in (1, 3, 4):
        a = rand_img(9, 13, cn, dt)
        for (t, b, l, r) in [(0, 0, 0, 0), (1, 2, 3, 4), (5, 5, 5, 5), (0, 7, 0, 11), (12, 1, 20, 0)]:
            for mode in ("constant", "replicate"):
                v = 0.0 if mode == "replicate" else 3.0
                out = cv2.copyMakeBorder(a, t, b, l, r, BORDERS[mode], value=(v, v, v, v))
                case(f"border_{np.dtype(dt).name}c{cn}_{t}{b}{l}{r}_{mode}", "copy_make_border",
                     {"top": t, "bottom": b, "left": l, "right": r, "border": mode, "value": v}, [a], out)
q = rng.random((3, 1)).astype(np.float32)
case("border_tiny_replicate", "copy_make_border", {"top": 4, "bottom": 2, "left": 3, "right": 5, "border": "replicate", "value": 0.0}, [q],
     cv2.copyMakeBorder(q, 4, 2, 3, 5, cv2.BORDER_REPLICATE))

# ---------------- pointPolygonTest ----------------
for t in range(40):
    if t < 10:  # behind.py's torso quads (float32)
        sh = rng.uniform(0, 600, (2, 2))
        poly = np.array([sh[0], sh[1], sh[1] + [rng.uniform(-30, 30), rng.uniform(50, 200)], sh[0] + [rng.uniform(-30, 30), rng.uniform(50, 200)]])
        poly = (poly.mean(0) + (poly - poly.mean(0)) * 1.15)
    elif t < 20:  # integer polygons (edge/vertex hits are exact)
        poly = rng.integers(0, 20, (rng.integers(3, 8), 2)).astype(np.float64)
    else:
        poly = rng.uniform(-50, 50, (rng.integers(3, 9), 2))
    poly = poly.astype(np.float32)
    pts = np.concatenate([rng.uniform(poly.min(0) - 5, poly.max(0) + 5, (60, 2)), poly.astype(np.float64),
                          (poly + np.roll(poly, -1, 0)).astype(np.float64) / 2, np.round(rng.uniform(poly.min(0) - 2, poly.max(0) + 2, (40, 2)))])
    res = [cv2.pointPolygonTest(poly, (float(x), float(y)), False) for x, y in pts]
    case(f"ppt_{t}", "point_polygon_test", {}, [poly, pts.astype(np.float64)], np.array(res, np.float64))

# ---------------- invertAffineTransform ----------------
for t in range(30):
    M = rng.normal(0, 3, (2, 3))
    if t % 5 == 0:
        M[:, :2] = 0  # singular
    dt = np.float32 if t % 2 else np.float64
    M = M.astype(dt)
    case(f"invaff_{t}_{np.dtype(dt).name}", "invert_affine", {}, [M], cv2.invertAffineTransform(M))

# ---------------- transform ----------------
Y2B = np.array([[1.1642, 2.0167, 0.0, -278.2338], [1.1645, -0.3897, -0.8121, 133.9585], [1.1642, 0.0, 1.5956, -224.693]], np.float32)
for (h, w) in [(1, 1), (3, 5), (238, 322), (476, 643)]:
    y = rng.integers(16, 236, (h, w), dtype=np.uint8)
    uvn = rng.integers(16, 241, (h, w, 2), dtype=np.uint8)
    x = np.dstack([y, uvn]).astype(np.float32)
    case(f"transform_yuv_{h}x{w}", "transform", {"m": [float(v) for v in Y2B.ravel()], "rows": 3, "cols": 4}, [x], cv2.transform(x, Y2B))
    M3 = rng.normal(0, 1, (3, 3)).astype(np.float32)
    case(f"transform_m3_{h}x{w}", "transform", {"m": [float(v) for v in M3.ravel()], "rows": 3, "cols": 3}, [x], cv2.transform(x, M3))

# ---------------- filter2D (lighting's motion blur on a padded premultiplied sprite) ----------------


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


for t, (vx, vy) in enumerate([(1.5, 0.3), (3.0, 1.0), (0.0, 4.6), (5.5, -2.0), (7.9, 3.3), (12.0, 0.0), (-9.0, 14.0)]):
    ker = motion_kernel(vx, vy)
    for s in (24, 77):
        sp = rng.random((s, s, 4)).astype(np.float32)
        pm = np.concatenate([sp[..., :3] * sp[..., 3:], sp[..., 3:]], -1)
        pad = ker.shape[0]
        pm = cv2.copyMakeBorder(pm, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        case(f"filter2d_motion{t}_k{ker.shape[0]}_s{s}", "filter2d", {"kw": ker.shape[1], "kh": ker.shape[0]}, [pm, ker], cv2.filter2D(pm, -1, ker))
    g = JG[0].astype(np.float32) / 255
    case(f"filter2d_motion{t}_k{ker.shape[0]}_gray", "filter2d", {"kw": ker.shape[1], "kh": ker.shape[0]}, [g, ker], cv2.filter2D(g, -1, ker))
for ks in (3, 5):
    ker = rng.random((ks, ks)).astype(np.float32)
    ker /= ker.sum()
    a = rand_img(31, 45, 3, np.float32)
    case(f"filter2d_rand_k{ks}", "filter2d", {"kw": ker.shape[1], "kh": ker.shape[0]}, [a, ker], cv2.filter2D(a, -1, ker))

# ---------------- edge cases ----------------
for dt in (np.uint8, np.float32):
    for cn in (1, 3):
        one = rand_img(1, 1, cn, dt)
        for inter in INTERS:
            rs_dsize("one", one, 5, 3, inter)
        row = rand_img(1, 17, cn, dt)
        for inter in INTERS:
            rs_dsize("row", row, 9, 4, inter)
            rs_dsize("row", row, 40, 1, inter)

# ---------------- 4K benchmark inputs (not cases) ----------------
big = cv2.resize(J[0], (3840, 2160), interpolation=cv2.INTER_CUBIC)
np.save(os.path.join(OUT, "bench_bgr4k.npy"), big)
np.save(os.path.join(OUT, "bench_gray4k.npy"), cv2.cvtColor(big, cv2.COLOR_BGR2GRAY))
np.save(os.path.join(OUT, "bench_grayf4k.npy"), cv2.cvtColor(big, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255)
np.save(os.path.join(OUT, "bench_bgrf4k.npy"), big.astype(np.float32))

json.dump({"cv2": cv2.__version__, "cases": cases}, open(os.path.join(OUT, "manifest.json"), "w"), indent=0)
print(json.dumps({"out": OUT, "cases": len(cases), "cv2": cv2.__version__}))
