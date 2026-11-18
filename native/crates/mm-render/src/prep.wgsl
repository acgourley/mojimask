// Decoded frame (stored size and orientation) -> the timeline's picture: resized to the display size with
// OpenCV's INTER_AREA weights (anamorphic pixels made square, like video_io) and turned upright (display matrix).
struct U { sw: u32, sh: u32, dw: u32, dh: u32, ow: u32, oh: u32, rot: u32, _p: u32, };
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> u: U;

// one axis of cv2's computeResizeAreaTab: source range and weights for destination index d
fn area(d: u32, s_len: u32, d_len: u32, out_w: ptr<function, array<f32, 8>>, out_i: ptr<function, array<i32, 8>>) -> i32 {
    if (s_len == d_len) { (*out_w)[0] = 1.0; (*out_i)[0] = i32(d); return 1; }
    let scale = f32(s_len) / f32(d_len);
    let f1 = f32(d) * scale; let f2 = f1 + scale;
    let s1 = i32(ceil(f1)); let s2 = i32(floor(f2));
    let cw = min(scale, f32(s_len) - f1);
    var k = 0;
    if (f32(s1) - f1 > 1e-3) { (*out_i)[k] = s1 - 1; (*out_w)[k] = (f32(s1) - f1) / cw; k++; }
    for (var s = s1; s < s2 && k < 7; s++) { (*out_i)[k] = s; (*out_w)[k] = 1.0 / cw; k++; }
    if (f2 - f32(s2) > 1e-3 && k < 8) { (*out_i)[k] = s2; (*out_w)[k] = min(min(f2 - f32(s2), 1.0), cw) / cw; k++; }
    return k;
}

@compute @workgroup_size(16, 16)
fn prep(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= u.ow || g.y >= u.oh) { return; }
    var x = g.x; var y = g.y;
    if (u.rot == 270u) { x = u.dw - 1u - g.y; y = g.x; }
    else if (u.rot == 90u) { x = g.y; y = u.dh - 1u - g.x; }
    else if (u.rot == 180u) { x = u.dw - 1u - g.x; y = u.dh - 1u - g.y; }
    var wx: array<f32, 8>; var ix: array<i32, 8>; var wy: array<f32, 8>; var iy: array<i32, 8>;
    let nx = area(x, u.sw, u.dw, &wx, &ix);
    let ny = area(y, u.sh, u.dh, &wy, &iy);
    var acc = vec3<f32>(0.0);
    for (var j = 0; j < ny; j++) {
        var row = vec3<f32>(0.0);
        for (var i = 0; i < nx; i++) {
            row += wx[i] * round(textureLoad(src, vec2<i32>(ix[i], iy[j]), 0).rgb * 255.0);
        }
        acc += wy[j] * row;
    }
    textureStore(dst, g.xy, vec4<f32>(round(acc) / 255.0, 1.0));
}
