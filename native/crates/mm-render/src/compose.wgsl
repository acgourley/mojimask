// WGSL port of mojimask/gpu_draw.py's Metal kernel ('guide' + 'compose'), same maths.
// Differences from the Metal source, all about storage, not maths:
//  - the frame is read from a texture (rgba8unorm or bgra8unorm; textureLoad gives byte/255 like uchar/255.0)
//    and written to a separate bgra8unorm storage texture (WGSL has no 8-bit storage buffers); untouched pixels
//    are copied through unchanged, so the result equals Metal's in-place write.
//  - the kernel works in BGR order like the Metal one (img is BGR there): we swizzle with .bgr on load/store.
//  - Metal's round() rounds halves away from zero, WGSL's to even: rnd() below reproduces Metal's.
// Additions (mm-core cover.rs documents the row):
//  - P[7] is a set of flag bits (1 lit, 2 posed, 4 mirrored); Python's rows hold 0 or 1 there, which read the same.
//  - posed covers (head pose drawn here, not on the CPU): P[34] roll (deg), P[35] yaw; the box is head_pose's canvas
//    (CANVAS x the sprite) and each canvas point is mapped back through the card turn to the base sprite (sprite_uv).

const STRIDE: u32 = 36u;
const FACE_DY: f32 = @FACE_DY@;
const PI: f32 = 3.14159265358979;

struct Frame {
    W: u32, H: u32, n: u32, mw: u32, mh: u32,
    kx: f32, ky: f32, seed: f32, soft: f32, ss: f32, sr: f32, z0: f32, z1: f32,
    // placement of the whole picture in the output (editor: 0,0 / 1)
    _p0: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> C: array<f32>;
@group(0) @binding(2) var<uniform> F: Frame;
@group(0) @binding(3) var<storage, read> nearm: array<f32>;
@group(0) @binding(4) var<storage, read_write> guide_buf: array<vec4<f32>>;
@group(0) @binding(5) var atlas: texture_2d_array<f32>;
@group(0) @binding(6) var samp: sampler;
@group(0) @binding(7) var dst: texture_storage_2d<bgra8unorm, write>;
// what the covers are drawn over: the frame itself (src), or the blur/pixelate/solid styles' picture of it
@group(0) @binding(8) var base: texture_2d<f32>;

fn rnd(x: f32) -> f32 { return select(-floor(-x + 0.5), floor(x + 0.5), x >= 0.0); }

fn to_lin(x: vec3<f32>) -> vec3<f32> {
    return select(pow((x + 0.055) / 1.055, vec3<f32>(2.4)), x / 12.92, x <= vec3<f32>(0.04045));
}
fn to_srgb(xi: vec3<f32>) -> vec3<f32> {
    let x = clamp(xi, vec3<f32>(0.0), vec3<f32>(1.0));
    return select(1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3<f32>(0.0031308));
}

fn pcg3d(vi: vec3<u32>) -> vec3<u32> {
    var v = vi * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> vec3<u32>(16u);
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}
fn gauss2(v: vec3<u32>) -> vec2<f32> {
    let h = pcg3d(v);
    let u1 = (f32(h.x) + 1.0) / 4294967296.0;
    let u2 = f32(h.y) / 4294967296.0;
    let r = sqrt(-2.0 * log(u1));
    return r * vec2<f32>(cos(2.0 * PI * u2), sin(2.0 * PI * u2));
}

fn P(b: u32, k: u32) -> f32 { return C[b + k]; }
fn flags(b: u32) -> u32 { return u32(max(P(b, 7u), 0.0) + 0.5); }
fn lit(b: u32) -> bool { return (flags(b) & 1u) != 0u; }

const CANVAS: f32 = 1.5;
const MIN_W: f32 = 0.8;

// Where loc (0..1 across the cover's box) samples the atlas. Unposed: the picture's top-left n x n corner. Posed: the
// box is head_pose.posed_sprite's canvas (S = CANVAS * n0 px); canvas pixel X (centres at integers) -> rotate back by
// roll -> the card turn's inverse (a projective map whose verticals stay vertical) -> sprite pixel (centres at
// integers) -> atlas coordinate. Outside the canvas or the sprite: (-1, -1), i.e. transparent.
fn sprite_uv(loc: vec2<f32>, b: u32) -> vec2<f32> {
    if ((flags(b) & 2u) == 0u) { return loc * P(b, 31u); }
    if (any(loc < vec2<f32>(0.0)) || any(loc > vec2<f32>(1.0))) { return vec2<f32>(-1.0); }
    let tex = f32(textureDimensions(atlas).x);
    let n0 = P(b, 31u) * tex;
    let S = rnd(n0 * CANVAS);
    let q = loc * S - 0.5 - S * 0.5;
    let th = P(b, 34u) * (PI / 180.0);
    let cs = cos(th); let sn = sin(th);
    let x = cs * q.x + sn * q.y;    // R^T q
    let y = -sn * q.x + cs * q.y;
    let yaw = P(b, 35u);
    let a = abs(yaw);
    let w = max(MIN_W, 1.0 - 0.25 * a) * n0 * 0.5;
    let nr = (1.0 + 0.06 * a) * n0 * 0.5;
    let fr = (1.0 - 0.18 * a) * n0 * 0.5;
    var lh = nr; var rh = fr;
    if (yaw > 0.0) { lh = fr; rh = nr; }
    let c = lh / rh - 1.0;
    let u = (x + w) / (w * (2.0 + c) - c * x);
    let v = 0.5 + y * (1.0 + c * u) / (2.0 * lh);
    var sx = u * n0;
    if ((flags(b) & 4u) != 0u) { sx = n0 - 1.0 - sx; }
    return (vec2<f32>(sx, v * n0) + 0.5) / tex;
}

// One sample of cover b's emoji at loc (0..1 across its square), lit; premultiplied BGRA.
fn shade(loc: vec2<f32>, b: u32) -> vec4<f32> {
    let v = textureSampleLevel(atlas, samp, sprite_uv(loc, b), i32(P(b, 3u)), P(b, 30u));
    if (v.a <= 1e-5) { return vec4<f32>(0.0); }
    let c = v.rgb / v.a;
    let sat = P(b, 18u);
    if (!lit(b) && sat == 1.0) { return v; }
    var alb = to_lin(c);
    if (sat != 1.0) {
        let L = dot(alb, vec3<f32>(0.0722, 0.7152, 0.2126));
        alb = max(L + sat * (alb - L), vec3<f32>(0.0));
    }
    var o: vec3<f32>;
    if (lit(b)) {
        let xy = loc * 2.0 - 1.0;
        let ramp = clamp(1.0 + P(b, 14u) * xy.x + P(b, 15u) * xy.y, 0.2, 2.0);
        let key = vec3<f32>(P(b, 8u), P(b, 9u), P(b, 10u));
        let amb = vec3<f32>(P(b, 11u), P(b, 12u), P(b, 13u));
        let lit = max(alb * (key * ramp + amb), alb * P(b, 17u));
        let m = (1.0 - P(b, 16u)) * alb + P(b, 16u) * lit;
        o = to_srgb(P(b, 19u) + m * (1.0 - P(b, 19u)));
    } else {
        o = to_srgb(alb);
    }
    return vec4<f32>(o * v.a, v.a);
}

fn near_at(p: vec2<f32>, g: vec3<f32>) -> f32 {
    let q = p * vec2<f32>(F.kx, F.ky) - 0.5;
    let b = vec2<i32>(floor(q));
    var acc = 0.0;
    var ws = 0.0;
    let hi = vec2<i32>(i32(F.mw) - 1, i32(F.mh) - 1);
    for (var dy = -1; dy <= 2; dy++) {
        for (var dx = -1; dx <= 2; dx++) {
            let u = b + vec2<i32>(dx, dy);
            let t = clamp(u, vec2<i32>(0), hi);
            let dd = vec2<f32>(u) - q;
            let gi = u32(t.y) * F.mw + u32(t.x);
            let dg = g - guide_buf[gi].rgb;
            let w = exp(-dot(dd, dd) / (2.0 * F.ss * F.ss) - dot(dg, dg) / (2.0 * F.sr * F.sr));
            acc += w * nearm[gi];
            ws += w;
        }
    }
    if (ws > 1e-8) { return acc / ws; }
    let t = clamp(vec2<i32>(i32(rnd(q.x)), i32(rnd(q.y))), vec2<i32>(0), hi);
    return nearm[u32(t.y) * F.mw + u32(t.x)];
}

fn img_bgr(x: u32, y: u32) -> vec3<f32> {  // the byte values (0..255) at x, y, BGR order
    return vec3<f32>(round(textureLoad(src, vec2<u32>(x, y), 0).bgr * 255.0));
}

@compute @workgroup_size(16, 16)
fn guide(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= F.mw || gid.y >= F.mh) { return; }
    let x0 = f32(gid.x) / F.kx; let x1 = f32(gid.x + 1u) / F.kx;
    let y0 = f32(gid.y) / F.ky; let y1 = f32(gid.y + 1u) / F.ky;
    let sx = max(1.0, (x1 - x0) / 8.0); let sy = max(1.0, (y1 - y0) / 8.0);
    var acc = vec3<f32>(0.0);
    var n = 0.0;
    for (var y = y0 + 0.5 * sy; y < y1; y += sy) {
        for (var x = x0 + 0.5 * sx; x < x1; x += sx) {
            acc += img_bgr(min(u32(x), F.W - 1u), min(u32(y), F.H - 1u));
            n += 1.0;
        }
    }
    guide_buf[gid.y * F.mw + gid.x] = vec4<f32>(acc / max(n, 1.0) / 255.0, 0.0);
}

@compute @workgroup_size(16, 16)
fn compose(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= F.W || gid.y >= F.H) { return; }
    let orig = textureLoad(src, gid.xy, 0).bgr;
    let base4 = textureLoad(base, gid.xy, 0);
    var col = base4.bgr;
    let p = vec2<f32>(gid.xy) + 0.5;
    var touched = false;
    var have_near = false;
    var nearp = 0.0;
    for (var i = 0u; i < F.n; i++) {
        let b = i * STRIDE;
        let size = P(b, 2u);
        let m = P(b, 29u) / size;
        let loc = (p - vec2<f32>(P(b, 0u), P(b, 1u))) / size;
        let in_sprite = all(loc >= vec2<f32>(-m)) && all(loc <= vec2<f32>(1.0 + m));
        let dc = p - vec2<f32>(P(b, 25u), P(b, 26u));
        let in_occ = P(b, 22u) > 0.5 && abs(dc.x) <= P(b, 27u) && abs(dc.y) <= P(b, 27u);
        if (!in_sprite && !in_occ) { continue; }
        touched = true;
        if (in_sprite) {
            var acc = vec4<f32>(0.0);
            let taps = i32(P(b, 6u));
            if (taps <= 1 && !lit(b)) {
                acc = shade(loc, b);
            } else if (taps <= 1) {
                let d = 0.5 / size;
                acc = 0.25 * (shade(loc + vec2<f32>(-d, -d), b) + shade(loc + vec2<f32>(d, -d), b)
                            + shade(loc + vec2<f32>(-d, d), b) + shade(loc + vec2<f32>(d, d), b));
            } else {
                let v = vec2<f32>(P(b, 4u), P(b, 5u));
                for (var k = 0; k < taps; k++) {
                    acc += shade(loc + v * (f32(k) / f32(taps - 1) - 0.5), b);
                }
                acc /= f32(taps);
            }
            if (P(b, 20u) > 0.0 && acc.a > 0.0) {
                let s8 = u32(F.seed) * 8u;
                let n1 = gauss2(vec3<u32>(gid.x, gid.y, s8 + 1u));
                let n2 = gauss2(vec3<u32>(gid.x, gid.y, s8 + 2u));
                let nz = n1.x * P(b, 20u) + P(b, 21u) * 0.5 * P(b, 20u) * vec3<f32>(n1.y, n2.x, n2.y);
                acc = vec4<f32>(acc.rgb + nz * acc.a, acc.a);
            }
            col = clamp(acc.rgb + (1.0 - acc.a) * col, vec3<f32>(0.0), vec3<f32>(1.0));
        }
        if (in_occ) {
            if (!have_near) { nearp = near_at(p, orig); have_near = true; }
            let r = length(p - vec2<f32>(P(b, 25u), P(b, 26u) + FACE_DY * P(b, 33u))) / max(P(b, 33u), 1e-6);
            let margin = mix(P(b, 24u), min(P(b, 32u), P(b, 24u)), clamp((r - F.z0) / (F.z1 - F.z0), 0.0, 1.0));
            var w = clamp((nearp - P(b, 23u) - margin) / F.soft + 0.5, 0.0, 1.0);
            if (w > 0.0) {
                for (var j = 0u; j < F.n; j++) {
                    if (j == i) { continue; }
                    let qb = j * STRIDE;
                    let q = p - 0.5 - vec2<f32>(P(qb, 25u), P(qb, 26u));
                    if (dot(q, q) <= P(qb, 28u) * P(qb, 28u)) { w = 0.0; break; }
                }
            }
            col = mix(col, orig, vec3<f32>(w));
        }
    }
    if (!touched) {
        textureStore(dst, gid.xy, vec4<f32>(base4.rgb, 1.0));
        return;
    }
    let c = clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)) * 255.0;
    let o = vec3<f32>(rnd(c.x), rnd(c.y), rnd(c.z)) / 255.0;
    textureStore(dst, gid.xy, vec4<f32>(o.zyx, 1.0));
}
