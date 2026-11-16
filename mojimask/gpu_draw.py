"""GPU drawing (Metal, through pyobjc): a frame's covers in one compute pass.

track.draw_frame hands its emoji/portrait drawing here when the GPU is there (available()), so the editor's frames and
Export get the same picture. Per pixel, walking the frame's covers far to near like draw_frame does:
  - the emoji: sampled from its picture on the GPU (mipmapped, so small covers don't shimmer), lit (lighting.py's
    lit_sprite maths), motion-blurred along the cover's movement, softened 0.6 px and given the footage's grain;
  - with MOJIMASK_DEPTH (depth.Occlusion), the part of the cover that is in front of the face shows the video again: the
    depth map (~256 px) is brought to full size with a joint bilateral upsample (Kopf et al. 2007) -- each pixel takes
    the nearby depth samples whose colour matches its own, so the cut follows the occluder's edge in the picture --
    and compared with the face's depth; never where another cover is drawn.
The CPU only prepares small per-cover numbers and uploads each emoji picture once (cached by picture).
"""
import collections
import math
import os
import struct
import threading

import cv2
import numpy as np

STRIDE = 36          # floats per cover in the covers buffer (layout: Compositor._cover)
TEX, SLICES = 384, 48  # emoji pictures live in a 2D texture array: up to SLICES pictures of up to TEX px
SIGMA_S, SIGMA_R = 0.9, 0.08  # joint bilateral upsample: spatial (depth-map texels), colour (0..1 BGR distance)

SRC = r"""
#include <metal_stdlib>
using namespace metal;

constant int STRIDE = @STRIDE@;
constant float FACE_DY = @FACE_DY@;  // track.FACE_DY: the face's centre below the head box's

struct Frame { uint W, H, n, mw, mh; float kx, ky, seed, soft, ss, sr, z0, z1; };

inline float3 to_lin(float3 x) { return select(pow((x + 0.055) / 1.055, 2.4), x / 12.92, x <= 0.04045); }
inline float3 to_srgb(float3 x) {
    x = clamp(x, 0.0, 1.0);
    return select(1.055 * pow(x, 1.0 / 2.4) - 0.055, x * 12.92, x <= 0.0031308);
}

inline uint3 pcg3d(uint3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}
inline float2 gauss2(uint3 v) {  // two independent N(0, 1)
    uint3 h = pcg3d(v);
    float u1 = (float(h.x) + 1.0) / 4294967296.0, u2 = float(h.y) / 4294967296.0;
    float r = sqrt(-2.0 * log(u1));
    return r * float2(cos(2.0 * M_PI_F * u2), sin(2.0 * M_PI_F * u2));
}

// One sample of cover P's emoji at loc (0..1 across its square), lit; premultiplied BGRA.
inline float4 shade(texture2d_array<float> atlas, float2 loc, device const float* P) {
    constexpr sampler s(address::clamp_to_zero, filter::linear, mip_filter::linear);
    float4 v = atlas.sample(s, loc * P[31], uint(P[3]), level(P[30]));
    if (v.a <= 1e-5) return float4(0.0);
    float3 c = v.rgb / v.a;
    float sat = P[18];
    if (P[7] < 0.5 && sat == 1.0) return v;  // flat, as drawn
    float3 alb = to_lin(c);
    if (sat != 1.0) {
        float L = dot(alb, float3(0.0722, 0.7152, 0.2126));
        alb = max(L + sat * (alb - L), 0.0);
    }
    float3 o;
    if (P[7] > 0.5) {
        float2 xy = loc * 2.0 - 1.0;
        float ramp = clamp(1.0 + P[14] * xy.x + P[15] * xy.y, 0.2, 2.0);
        float3 key = float3(P[8], P[9], P[10]), amb = float3(P[11], P[12], P[13]);
        float3 lit = max(alb * (key * ramp + amb), alb * P[17]);
        float3 m = (1.0 - P[16]) * alb + P[16] * lit;
        o = to_srgb(P[19] + m * (1.0 - P[19]));
    } else {
        o = to_srgb(alb);
    }
    return float4(o * v.a, v.a);
}

// The depth map's nearness at pixel p, brought to full size along the picture's edges.
inline float near_at(float2 p, float3 g, constant Frame& F, device const float* nearm, device const float4* guide) {
    float2 q = p * float2(F.kx, F.ky) - 0.5;
    int2 b = int2(floor(q));
    float acc = 0.0, ws = 0.0;
    for (int dy = -1; dy <= 2; dy++) {
        for (int dx = -1; dx <= 2; dx++) {
            int2 u = b + int2(dx, dy);
            int2 t = clamp(u, int2(0), int2(int(F.mw) - 1, int(F.mh) - 1));
            float2 dd = float2(u) - q;
            float3 dg = g - guide[t.y * F.mw + t.x].rgb;
            float w = exp(-dot(dd, dd) / (2.0 * F.ss * F.ss) - dot(dg, dg) / (2.0 * F.sr * F.sr));
            acc += w * nearm[t.y * F.mw + t.x];
            ws += w;
        }
    }
    if (ws > 1e-8) return acc / ws;
    int2 t = clamp(int2(round(q)), int2(0), int2(int(F.mw) - 1, int(F.mh) - 1));
    return nearm[t.y * F.mw + t.x];
}

// The frame's colour at depth-map resolution (the guide for near_at).
kernel void guide(device const uchar* img [[buffer(0)]], constant Frame& F [[buffer(2)]],
                  device float4* out [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= F.mw || gid.y >= F.mh) return;
    float x0 = gid.x / F.kx, x1 = (gid.x + 1) / F.kx, y0 = gid.y / F.ky, y1 = (gid.y + 1) / F.ky;
    float sx = max(1.0, (x1 - x0) / 8.0), sy = max(1.0, (y1 - y0) / 8.0);
    float3 acc = 0.0; float n = 0.0;
    for (float y = y0 + 0.5 * sy; y < y1; y += sy) {
        for (float x = x0 + 0.5 * sx; x < x1; x += sx) {
            uint i = (min(uint(y), F.H - 1) * F.W + min(uint(x), F.W - 1)) * 3;
            acc += float3(img[i], img[i + 1], img[i + 2]);
            n += 1.0;
        }
    }
    out[gid.y * F.mw + gid.x] = float4(acc / max(n, 1.0) / 255.0, 0.0);
}

kernel void compose(device uchar* img [[buffer(0)]], device const float* C [[buffer(1)]],
                    constant Frame& F [[buffer(2)]], device const float* nearm [[buffer(3)]],
                    device const float4* guide [[buffer(4)]], texture2d_array<float> atlas [[texture(0)]],
                    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= F.W || gid.y >= F.H) return;
    uint idx = (gid.y * F.W + gid.x) * 3;
    float3 orig = float3(img[idx], img[idx + 1], img[idx + 2]) / 255.0;
    float3 col = orig;
    float2 p = float2(gid) + 0.5;
    bool touched = false, have_near = false;
    float nearp = 0.0;
    for (uint i = 0; i < F.n; i++) {
        device const float* P = C + i * STRIDE;
        float size = P[2], m = P[29] / size;
        float2 loc = (p - float2(P[0], P[1])) / size;
        bool in_sprite = all(loc >= -m) && all(loc <= 1.0 + m);
        float2 dc = p - float2(P[25], P[26]);
        bool in_occ = P[22] > 0.5 && abs(dc.x) <= P[27] && abs(dc.y) <= P[27];
        if (!in_sprite && !in_occ) continue;
        touched = true;
        if (in_sprite) {
            float4 acc = 0.0;
            int taps = int(P[6]);
            if (taps <= 1 && P[7] < 0.5) {  // flat: as track.paste draws it
                acc = shade(atlas, loc, P);
            } else if (taps <= 1) {  // lit: lighting's 0.6 px softening (sprites are sharper than phone video)
                float d = 0.5 / size;
                acc = 0.25 * (shade(atlas, loc + float2(-d, -d), P) + shade(atlas, loc + float2(d, -d), P)
                            + shade(atlas, loc + float2(-d, d), P) + shade(atlas, loc + float2(d, d), P));
            } else {  // motion blur: along the cover's movement over the shutter
                float2 v = float2(P[4], P[5]);
                for (int k = 0; k < taps; k++) acc += shade(atlas, loc + v * (float(k) / float(taps - 1) - 0.5), P);
                acc /= float(taps);
            }
            if (P[20] > 0.0 && acc.a > 0.0) {  // the footage's grain: luminance, plus colour as far as it has colour
                float2 n1 = gauss2(uint3(gid.x, gid.y, uint(F.seed) * 8u + 1u));
                float2 n2 = gauss2(uint3(gid.x, gid.y, uint(F.seed) * 8u + 2u));
                float3 n = n1.x * P[20] + P[21] * 0.5 * P[20] * float3(n1.y, n2.x, n2.y);
                acc.rgb += n * acc.a;
            }
            col = clamp(acc.rgb + (1.0 - acc.a) * col, 0.0, 1.0);
        }
        if (in_occ) {
            if (!have_near) { nearp = near_at(p, orig, F, nearm, guide); have_near = true; }
            // the face itself needs the face margin (P[24]); its rim only the model's "apart" (P[32]): depth.margin_at
            float r = length(p - float2(P[25], P[26] + FACE_DY * P[33])) / max(P[33], 1e-6);
            float margin = mix(P[24], min(P[32], P[24]), clamp((r - F.z0) / (F.z1 - F.z0), 0.0, 1.0));
            float w = clamp((nearp - P[23] - margin) / F.soft + 0.5, 0.0, 1.0);
            if (w > 0.0) {
                for (uint j = 0; j < F.n; j++) {  // never under another cover: a nearer face stays under its own emoji
                    if (j == i) continue;
                    device const float* Q = C + j * STRIDE;
                    float2 q = p - 0.5 - float2(Q[25], Q[26]);
                    if (dot(q, q) <= Q[28] * Q[28]) { w = 0.0; break; }
                }
            }
            col = mix(col, orig, w);
        }
    }
    if (!touched) return;
    float3 o = round(clamp(col, 0.0, 1.0) * 255.0);
    img[idx] = uchar(o.x); img[idx + 1] = uchar(o.y); img[idx + 2] = uchar(o.z);
}
"""


class Compositor:
    """draw(f, covers, near=None, seed=0) -> f with the covers drawn (a new array). covers: dicts from
    frame_covers(); near: the depth map's nearness for this frame (float32, the map's own size), if any cover has
    occlusion. Thread-safe (one frame on the GPU at a time; a frame takes a few ms)."""

    def __init__(self):
        import Metal
        from mojimask import track as T
        self.M = Metal
        self.dev = Metal.MTLCreateSystemDefaultDevice()
        if self.dev is None:
            raise RuntimeError("no Metal device")
        src = SRC.replace("@STRIDE@", str(STRIDE)).replace("@FACE_DY@", repr(float(T.FACE_DY)))
        lib, err = self.dev.newLibraryWithSource_options_error_(src, None, None)
        if lib is None:
            raise RuntimeError(f"shader didn't compile: {err}")
        self.pso = {}
        for n in ("compose", "guide"):
            pso, err = self.dev.newComputePipelineStateWithFunction_error_(lib.newFunctionWithName_(n), None)
            if pso is None:
                raise RuntimeError(f"pipeline {n}: {err}")
            self.pso[n] = pso
        self.q = self.dev.newCommandQueue()
        d = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            Metal.MTLPixelFormatRGBA16Float, TEX, TEX, True)
        d.setTextureType_(Metal.MTLTextureType2DArray)
        d.setArrayLength_(SLICES)
        d.setUsage_(Metal.MTLTextureUsageShaderRead)
        self.atlas = self.dev.newTextureWithDescriptor_(d)
        self.levels = self.atlas.mipmapLevelCount()
        self.slots = collections.OrderedDict()  # picture key -> (slice, size, picture kept alive so its id holds)
        self.lock = threading.Lock()
        self._img = (0, None, None)  # (bytes, MTLBuffer, numpy view)
        self._dirty = False

    # ---------------------------------------------------------------- pictures
    def slot(self, pic, pinned):
        """The atlas slice holding picture pic (RGBA uint8, RGB order, square), uploading it if needed."""
        key = (id(pic), pic.shape)
        if key in self.slots:
            self.slots.move_to_end(key)
            return self.slots[key][:2]
        if len(self.slots) < SLICES:
            sl = len(self.slots)
        else:
            old = next((k for k in self.slots if self.slots[k][0] not in pinned), None)
            if old is None:
                return None
            sl = self.slots.pop(old)[0]
        sp = pic if pic.shape[0] <= TEX else cv2.resize(pic, (TEX, TEX), interpolation=cv2.INTER_AREA)
        n = sp.shape[0]
        a = sp[..., 3:4].astype(np.float32) / 255
        pm = np.zeros((TEX, TEX, 4), np.float16)
        pm[:n, :n, :3] = sp[..., 2::-1].astype(np.float32) / 255 * a  # BGR, premultiplied
        pm[:n, :n, 3:] = a
        self.atlas.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage_(
            self.M.MTLRegionMake2D(0, 0, TEX, TEX), 0, sl, pm.tobytes(), TEX * 8, TEX * TEX * 8)
        self.slots[key] = (sl, n, pic)
        self._dirty = True
        return sl, n

    # ---------------------------------------------------------------- drawing
    def _cover(self, cv, sl, n):
        P = np.zeros(STRIDE, np.float32)
        x0, y0, s = cv["box"]
        P[0:3] = x0, y0, s
        P[3] = sl
        L = cv.get("vel")
        if L is not None:
            vx, vy = L
            ln = math.hypot(vx, vy)
            if ln >= 1.0:
                P[4:6] = vx / s, vy / s
                P[6] = min(64, max(3, int(math.ceil(ln)) * 2 + 1))
                P[29] = ln / 2 + 1
        P[29] = max(P[29], 1.0)
        lt = cv.get("light")
        if lt is not None:
            P[7] = 1
            P[8:11], P[11:14], P[14:16] = lt["key"], lt["amb"], lt["grad"]
            P[16], P[17], P[19] = lt["mix"], lt["floor"], lt["black"]
            P[20], P[21] = lt["grain"] / 255.0, lt["colour_grain"]
        P[18] = cv.get("saturation", 1.0)
        oc = cv.get("occ")
        P[25:27] = cv["centre"]
        P[27] = cv["half"]
        P[28] = cv["radius"]
        if oc is not None:
            P[22], P[23], P[24] = 1, oc["face"], oc["margin"]
            P[32], P[33] = oc["rim"], oc["head"]
        P[30] = max(0.0, math.log2(n / s)) if s > 0 else 0.0
        P[31] = n / TEX
        return P

    def _buf(self, nbytes):
        if self._img[0] < nbytes:
            b = self.dev.newBufferWithLength_options_(nbytes, self.M.MTLResourceStorageModeShared)
            self._img = (nbytes, b, np.frombuffer(b.contents().as_buffer(nbytes), np.uint8))
        return self._img[1], self._img[2][:nbytes]

    def draw(self, f, covers, near=None, seed=0, soft=0.15):
        H, W = f.shape[:2]
        if not covers:
            return f.copy()
        M = self.M
        with self.lock:
            pinned, rows = set(), []
            for cv in covers:
                got = self.slot(cv["pic"], pinned)
                if got is None:
                    return None  # more pictures in one frame than the atlas holds: caller draws on the CPU
                pinned.add(got[0])
                rows.append(self._cover(cv, *got))
            C = np.ascontiguousarray(np.stack(rows))
            buf, view = self._buf(H * W * 3)
            view[:] = np.ascontiguousarray(f).reshape(-1)
            if near is not None:
                mh, mw = near.shape
                nm = np.ascontiguousarray(near, np.float32)
            else:
                mh = mw = 1
                nm = np.zeros(1, np.float32)
            from mojimask import depth as DP
            fr = struct.pack("5I8f", W, H, len(rows), mw, mh, mw / W, mh / H, float(seed), soft, SIGMA_S, SIGMA_R,
                             *DP.FACE_ZONE)
            cbuf = self.dev.newBufferWithBytes_length_options_(C.tobytes(), C.nbytes, M.MTLResourceStorageModeShared)
            nbuf = self.dev.newBufferWithBytes_length_options_(nm.tobytes(), nm.nbytes, M.MTLResourceStorageModeShared)
            gbuf = self.dev.newBufferWithLength_options_(mw * mh * 16, M.MTLResourceStorageModeShared)
            cb = self.q.commandBuffer()
            if self._dirty:
                bl = cb.blitCommandEncoder()
                bl.generateMipmapsForTexture_(self.atlas)
                bl.endEncoding()
                self._dirty = False
            enc = cb.computeCommandEncoder()
            enc.setBuffer_offset_atIndex_(buf, 0, 0)
            enc.setBuffer_offset_atIndex_(cbuf, 0, 1)
            enc.setBytes_length_atIndex_(fr, len(fr), 2)
            enc.setBuffer_offset_atIndex_(nbuf, 0, 3)
            enc.setBuffer_offset_atIndex_(gbuf, 0, 4)
            enc.setTexture_atIndex_(self.atlas, 0)
            if near is not None:
                enc.setComputePipelineState_(self.pso["guide"])
                enc.dispatchThreads_threadsPerThreadgroup_(M.MTLSizeMake(mw, mh, 1), M.MTLSizeMake(16, 16, 1))
                enc.memoryBarrierWithScope_(M.MTLBarrierScopeBuffers)
            enc.setComputePipelineState_(self.pso["compose"])
            enc.dispatchThreads_threadsPerThreadgroup_(M.MTLSizeMake(W, H, 1), M.MTLSizeMake(16, 16, 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if cb.error() is not None:
                raise RuntimeError(f"GPU drawing failed: {cb.error()}")
            return view.reshape(H, W, 3).copy()


# -------------------------------------------------------------------- draw_frame's emoji path, on the GPU
def frame_covers(f, t, items, sprites, scale, overrides, shown, light, style, head_pose, portrait, saturation,
                 occlusion):
    """track.draw_frame's emoji/portrait path as a list of covers for Compositor.draw (+ the near map)."""
    from mojimask import track as T
    items = [x for x in T.draw_order(items, t, occlusion) if not (x[3:] and x[3] in shown)]
    near, occ = occlusion.plan(t, items, scale, f.shape) if occlusion is not None else (None, {})
    covers = []
    for i, (tid, c, kind, *rest) in enumerate(items):
        piece = rest[0] if rest else tid
        sp, cp = T.cover_picture(tid, piece, c, t, sprites, scale, overrides, style, head_pose, portrait)
        s = int(round(cp[2] * scale))
        cv = {"pic": sp, "centre": (c[0], c[1]), "half": c[2] * scale / 2, "radius": c[2] * scale / 2,
              "occ": occ.get(i)}
        if s < 4:
            cv["box"] = (-1e6, -1e6, 1.0)  # too small to draw (as on the CPU), still counts as a cover
        else:
            cv["box"] = (int(round(cp[0] - s / 2)), int(round(cp[1] - s / 2)), s)
        if light is not None:
            lp = light.params(piece, t, f)
            cv["light"] = {"key": lp["key"], "amb": lp["amb"], "grad": lp["grad"], "mix": lp["mix"],
                           "floor": lp["floor"], "black": lp["black"], "grain": lp["grain"],
                           "colour_grain": min(lp["colourfulness"], 1.0)}
            cv["saturation"] = lp["saturation"]
            if lp["vel"] is not None:
                vx, vy = lp["vel"]
                cv["vel"] = (vx * lp["shutter"], vy * lp["shutter"])
        else:
            cv["saturation"] = saturation
        covers.append(cv)
    return covers, near


_GPU = None
_GPU_LOCK = threading.Lock()


def get():
    """The shared Compositor, or None (no Metal, or MOJIMASK_GPU=0)."""
    global _GPU
    if os.environ.get("MOJIMASK_GPU", "1") == "0":
        return None
    with _GPU_LOCK:
        if _GPU is None:
            try:
                _GPU = Compositor()
            except Exception as e:  # noqa: BLE001 -- no Metal / no pyobjc Metal: draw on the CPU
                print(f"[gpu] off: {e}", flush=True)
                _GPU = False
        return _GPU or None


def draw_frame(f, t, items, sprites, scale=1.25, overrides=None, shown=None, light=None, style="emoji",
               head_pose=None, portrait=None, saturation=1.0, occlusion=None):
    """track.draw_frame's emoji/portrait path on the GPU; None if it can't (caller draws on the CPU)."""
    g = get()
    if g is None:
        return None
    covers, near = frame_covers(f, t, items, sprites, scale, overrides or {}, shown or set(), light, style,
                                head_pose, portrait, saturation, occlusion)
    from mojimask import depth as DP
    return g.draw(f, covers, near, seed=t, soft=DP.OCC_SOFT)
