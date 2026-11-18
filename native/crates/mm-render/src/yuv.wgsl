// Bi-planar YUV 4:2:0 (NV12 or P010, the decoder's own planes) -> RGB, written as rgba8 at the stored size.
// Matrix and range from the stream's colorimetry, or (use_m) a 3x4 affine matrix on 8-bit-unit codes;
// chroma upsampled bilinearly at its siting.
// transfer 1 = HLG (BT.2100) shown on an SDR BT.709 display (see hlg_to_sdr).
struct U {
    w: u32, h: u32, cw: u32, ch: u32,
    depth: u32, full: u32, hcos: u32, vcos: u32,
    kr: f32, kb: f32, transfer: u32, use_m: u32,
    // rows R, G, B of the 3x4 matrix on 8-bit-unit codes (Y, Cb, Cr, 1); used when use_m == 1
    m0: vec4<f32>, m1: vec4<f32>, m2: vec4<f32>,
};
@group(0) @binding(0) var ytex: texture_2d<f32>;
@group(0) @binding(1) var uvtex: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> u: U;

fn code(v: f32) -> f32 {
    if (u.depth == 8u) { return round(v * 255.0); }
    return round(v * 65535.0) / 64.0;   // P010: 10 bits in the high bits of 16
}

fn uv_at(x: i32, y: i32) -> vec2<f32> {
    let c = vec2<i32>(clamp(x, 0, i32(u.cw) - 1), clamp(y, 0, i32(u.ch) - 1));
    let v = textureLoad(uvtex, c, 0).rg;
    return vec2<f32>(code(v.x), code(v.y));
}

fn hlg_inv_oetf(e: f32) -> f32 {
    let a = 0.17883277; let b = 0.28466892; let c = 0.55991073;
    if (e <= 0.5) { return e * e / 3.0; }
    return (exp((e - c) / a) + b) / 12.0;
}

// HLG signal (BT.2020 primaries) -> scene light -> OOTF (gamma 1.2, nominal 1000 nit display) -> scaled so HLG
// reference white (75% signal) maps to SDR white -> BT.2020->BT.709 primaries -> BT.709 inverse EOTF (gamma 2.4 approx
// via BT.1886). A simple, documented choice; not AVFoundation's (unpublished) one.
fn hlg_to_sdr(rgb: vec3<f32>) -> vec3<f32> {
    let e = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let s = vec3<f32>(hlg_inv_oetf(e.r), hlg_inv_oetf(e.g), hlg_inv_oetf(e.b));
    let ys = 0.2627 * s.r + 0.6780 * s.g + 0.0593 * s.b;
    let disp = s * pow(max(ys, 1e-6), 0.2);              // OOTF, gamma 1.2 (relative, peak 1)
    let refw = pow(hlg_inv_oetf(0.75), 1.2);             // HLG 75% = diffuse white
    let lin2020 = disp / refw;
    let m = mat3x3<f32>(
        vec3<f32>(1.6605, -0.1246, -0.0182),
        vec3<f32>(-0.5876, 1.1329, -0.1006),
        vec3<f32>(-0.0728, -0.0083, 1.1187));
    let lin709 = clamp(m * lin2020, vec3<f32>(0.0), vec3<f32>(1.0));
    return pow(lin709, vec3<f32>(1.0 / 2.4));
}

@compute @workgroup_size(16, 16)
fn yuv(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= u.w || g.y >= u.h) { return; }
    let yc = code(textureLoad(ytex, vec2<i32>(g.xy), 0).r);
    var cx = (f32(g.x) - 0.5) * 0.5;
    if (u.hcos == 1u) { cx = f32(g.x) * 0.5; }
    var cy = (f32(g.y) - 0.5) * 0.5;
    if (u.vcos == 1u) { cy = f32(g.y) * 0.5; }
    let x0 = i32(floor(cx)); let y0 = i32(floor(cy));
    let fx = cx - floor(cx); let fy = cy - floor(cy);
    let c = mix(mix(uv_at(x0, y0), uv_at(x0 + 1, y0), fx), mix(uv_at(x0, y0 + 1), uv_at(x0 + 1, y0 + 1), fx), fy);
    var sc = 1.0;
    if (u.depth != 8u) { sc = 4.0; }
    if (u.use_m == 1u) {
        // AVFoundation's conversion (mm_media::convert::display_matrix): codes in 8-bit units, rounded half up
        let v = vec4<f32>(yc / sc, c.x / sc, c.y / sc, 1.0);
        let o = clamp(floor(vec3<f32>(dot(u.m0, v), dot(u.m1, v), dot(u.m2, v)) + 0.5), vec3<f32>(0.0), vec3<f32>(255.0));
        textureStore(dst, vec2<i32>(g.xy), vec4<f32>(o / 255.0, 1.0));
        return;
    }
    var yy: f32; var cb: f32; var cr: f32;
    if (u.full == 1u) {
        let m = 255.0 * sc;
        yy = yc / m; cb = (c.x - 128.0 * sc) / m; cr = (c.y - 128.0 * sc) / m;
    } else {
        yy = (yc - 16.0 * sc) / (219.0 * sc); cb = (c.x - 128.0 * sc) / (224.0 * sc); cr = (c.y - 128.0 * sc) / (224.0 * sc);
    }
    let kg = 1.0 - u.kr - u.kb;
    let r = yy + 2.0 * (1.0 - u.kr) * cr;
    let b = yy + 2.0 * (1.0 - u.kb) * cb;
    let gg = (yy - u.kr * r - u.kb * b) / kg;
    var rgb = clamp(vec3<f32>(r, gg, b), vec3<f32>(0.0), vec3<f32>(1.0));
    if (u.transfer == 1u) { rgb = hlg_to_sdr(rgb); }
    textureStore(dst, vec2<i32>(g.xy), vec4<f32>(round(rgb * 255.0) / 255.0, 1.0));
}
