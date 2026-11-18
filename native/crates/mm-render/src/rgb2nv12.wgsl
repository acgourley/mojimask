// Composite (bgra8, read as rgb) -> NV12 bytes, BT.709 limited range, for the readback export path.
// One invocation = a 4x2 luma block: two packed u32 of Y, one packed u32 of CbCr (2 pairs, from 2x2 averages).
struct U { w: u32, h: u32, _a: u32, _b: u32, };
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> u: U;

fn luma(c: vec3<f32>) -> f32 { return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b; }
fn b8(v: f32) -> u32 { return u32(clamp(round(v), 0.0, 255.0)); }

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let x0 = g.x * 4u; let y0 = g.y * 2u;
    if (x0 >= u.w || y0 >= u.h) { return; }
    var rows: array<u32, 2>;
    var cbcr = 0u;
    for (var j = 0u; j < 2u; j++) {
        var word = 0u;
        for (var i = 0u; i < 4u; i++) {
            let c = textureLoad(src, vec2<u32>(x0 + i, y0 + j), 0).rgb;
            word |= b8(16.0 + 219.0 * luma(c)) << (8u * i);
        }
        rows[j] = word;
    }
    for (var k = 0u; k < 2u; k++) {
        var acc = vec3<f32>(0.0);
        for (var j = 0u; j < 2u; j++) { for (var i = 0u; i < 2u; i++) {
            acc += textureLoad(src, vec2<u32>(x0 + 2u * k + i, y0 + j), 0).rgb;
        } }
        let c = acc * 0.25;
        let y = luma(c);
        let cb = 128.0 + 224.0 * (c.b - y) / 1.8556;
        let cr = 128.0 + 224.0 * (c.r - y) / 1.5748;
        cbcr |= (b8(cb) << (16u * k)) | (b8(cr) << (16u * k + 8u));
    }
    out[(y0 * u.w + x0) / 4u] = rows[0];
    out[((y0 + 1u) * u.w + x0) / 4u] = rows[1];
    out[(u.w * u.h + (y0 / 2u) * u.w + x0) / 4u] = cbcr;
}
