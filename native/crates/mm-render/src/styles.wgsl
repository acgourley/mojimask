// The CPU cover styles of mojimask/track.py cover() (blur, pixelate, solid) on the GPU, one circle at a time
// (circles are drawn in order, each over the result of the last, like the CPU loop).
struct U {
    x1: i32, y1: i32, x2: i32, y2: i32,
    cx: f32, cy: f32, r: f32, mode: u32,   // mode: 0 blur, 1 pixelate, 2 solid
    k: i32, bw: i32, bh: i32, _p: i32,     // blur kernel size; pixelate blocks across/down
};
@group(0) @binding(0) var a: texture_2d<f32>;
@group(0) @binding(1) var b: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> u: U;
@group(0) @binding(3) var<storage, read_write> s1: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> s2: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> kw: array<f32>;

fn rw() -> i32 { return u.x2 - u.x1; }
fn rh() -> i32 { return u.y2 - u.y1; }
fn refl(i: i32, n: i32) -> i32 {  // BORDER_REFLECT_101
    if (n == 1) { return 0; }
    var j = i;
    if (j < 0) { j = -j; }
    if (j >= n) { j = 2 * n - 2 - j; }
    return clamp(j, 0, n - 1);
}
fn rnd8(v: vec4<f32>) -> vec4<f32> { return round(clamp(v, vec4<f32>(0.0), vec4<f32>(255.0))); }
fn px(x: i32, y: i32) -> vec4<f32> { return round(textureLoad(a, vec2<i32>(u.x1 + x, u.y1 + y), 0) * 255.0); }

// blur pass 1 (from the picture) / 3 (from s2): horizontal into s1
@compute @workgroup_size(16, 16)
fn blur_h_src(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = i32(g.x); let y = i32(g.y);
    if (x >= rw() || y >= rh()) { return; }
    var acc = vec4<f32>(0.0);
    let h = u.k / 2;
    for (var i = -h; i <= h; i++) { acc += kw[i + h] * px(refl(x + i, rw()), y); }
    s1[y * rw() + x] = acc;
}
@compute @workgroup_size(16, 16)
fn blur_h_s2(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = i32(g.x); let y = i32(g.y);
    if (x >= rw() || y >= rh()) { return; }
    var acc = vec4<f32>(0.0);
    let h = u.k / 2;
    for (var i = -h; i <= h; i++) { acc += kw[i + h] * s2[y * rw() + refl(x + i, rw())]; }
    s1[y * rw() + x] = acc;
}
// vertical from s1 into s2, rounded to 8 bits (each GaussianBlur call returns uint8)
@compute @workgroup_size(16, 16)
fn blur_v(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = i32(g.x); let y = i32(g.y);
    if (x >= rw() || y >= rh()) { return; }
    var acc = vec4<f32>(0.0);
    let h = u.k / 2;
    for (var i = -h; i <= h; i++) { acc += kw[i + h] * s1[refl(y + i, rh()) * rw() + x]; }
    s2[y * rw() + x] = rnd8(acc);
}
// pixelate: INTER_AREA down to bw x bh blocks (area-weighted), into s2[0..bw*bh]
@compute @workgroup_size(8, 8)
fn pix_down(@builtin(global_invocation_id) g: vec3<u32>) {
    let bx = i32(g.x); let by = i32(g.y);
    if (bx >= u.bw || by >= u.bh) { return; }
    let fx = f32(rw()) / f32(u.bw); let fy = f32(rh()) / f32(u.bh);
    let x0 = f32(bx) * fx; let x1 = x0 + fx; let y0 = f32(by) * fy; let y1 = y0 + fy;
    var acc = vec4<f32>(0.0);
    var ws = 0.0;
    for (var y = i32(floor(y0)); y < i32(ceil(y1)) && y < rh(); y++) {
        let wy = min(y1, f32(y + 1)) - max(y0, f32(y));
        for (var x = i32(floor(x0)); x < i32(ceil(x1)) && x < rw(); x++) {
            let wx = min(x1, f32(x + 1)) - max(x0, f32(x));
            acc += wx * wy * px(x, y);
            ws += wx * wy;
        }
    }
    s2[by * u.bw + bx] = rnd8(acc / ws);
}
// the circle: new pixels inside it, the picture outside; the whole frame is copied to b beforehand
@compute @workgroup_size(16, 16)
fn combine(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = i32(g.x); let y = i32(g.y);
    if (x >= rw() || y >= rh()) { return; }
    let X = u.x1 + x; let Y = u.y1 + y;
    let dx = f32(X) - u.cx; let dy = f32(Y) - u.cy;
    var v = textureLoad(a, vec2<i32>(X, Y), 0);
    if (dx * dx + dy * dy <= u.r * u.r) {
        if (u.mode == 0u) {
            v = s2[y * rw() + x] / 255.0;
        } else if (u.mode == 1u) {
            let bx = min(i32(floor(f32(x) * (f32(u.bw) / f32(rw())) + 1e-4)), u.bw - 1);
            let by = min(i32(floor(f32(y) * (f32(u.bh) / f32(rh())) + 1e-4)), u.bh - 1);
            v = s2[by * u.bw + bx] / 255.0;
        } else {
            v = vec4<f32>(40.0 / 255.0, 40.0 / 255.0, 40.0 / 255.0, 1.0);
        }
    }
    textureStore(b, vec2<i32>(X, Y), vec4<f32>(v.rgb, 1.0));
}
