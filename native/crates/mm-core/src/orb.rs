//! The "orb" cover (developer style, an experiment towards 3-D emoji): a little 3-D head, a sphere with eyes, nose, a
//! smile and ears, turned with the person's head in 3-D.
//!
//! The head model (MODEL, HEAD_R, in interocular distances) and the rotation are shared by the fit
//! (mm_analysis::head3d, from the pose skeleton's nose, eyes and ears) and the drawing (`picture`, here).
//!
//! Drawing: each cover's picture is the orb turned by yaw and pitch, rounded to STEP degrees and keyed
//! `orb:<yaw>:<pitch>` ([`key`]), so one picture serves every head turned that way; the shader tilts it by roll (the
//! row's posed flag with yaw 0: no card turn, no sideways shift) and lights it like an emoji (key light, ambient, the
//! brightness ramp, motion blur, grain, depth cut). The picture carries the sphere's own shading for a light from
//! the camera (wrapped Lambert), which the shader's ramp then tips towards the brighter side of the face.
//!
//! Conventions (camera space): x image right, y image down, z towards the camera; yaw + = face points image right,
//! pitch + = face points down, roll + = clockwise in the image (head_pose.rs's roll). R = Rz(roll) Rx(pitch) Ry(yaw).

/// radius of the head sphere, in interocular distances (head ~18 cm wide, eyes ~6.3 cm apart)
pub const HEAD_R: f64 = 1.35;

/// Head model, in interocular distances, head centre at the origin: [nose, left eye, right eye, left ear, right ear]
/// (COCO order 0..4). The model's +x side is the subject's left (COCO "left eye"), seen on image right when facing
/// the camera. Rough adult proportions, not measured.
pub const MODEL: [[f64; 3]; 5] = [
    [0.0, 0.45, 1.40],   // nose tip: in front of the sphere, a little below the eyes
    [0.5, -0.05, 1.05],  // left eye
    [-0.5, -0.05, 1.05], // right eye
    [1.25, 0.15, -0.15], // left ear: at the side, just behind the centre, a little below the eyes
    [-1.25, 0.15, -0.15],
];

/// the ears as drawn: spheres of EAR_R at (+-EAR_AT, the model's ear height and depth)
pub const EAR_R: f64 = 0.32;
pub const EAR_AT: f64 = 1.22;
/// the picture's half side, in head-sphere radii: room for the ears however the head turns (an ear's far edge is
/// sqrt(EAR_AT^2 + 0.15^2 + 0.15^2) + EAR_R = 1.56 model units = 1.15 radii from the centre)
pub const HALF: f64 = 1.2;
/// picture key rounding, degrees
pub const STEP: f64 = 10.0;
pub const MAX_YAW: f64 = 90.0;
pub const MAX_PITCH: f64 = 40.0;
pub const PREFIX: &str = "orb:";

pub type Mat3 = [[f64; 3]; 3];

/// R = Rz(roll) Rx(pitch) Ry(yaw), radians
pub fn rot(yaw: f64, pitch: f64, roll: f64) -> Mat3 {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sr, cr) = roll.sin_cos();
    let ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
    let rx = [[1.0, 0.0, 0.0], [0.0, cp, sp], [0.0, -sp, cp]];
    let rz = [[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]];
    mul(&rz, &mul(&rx, &ry))
}

pub fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    m
}

pub fn apply(m: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

/// m transposed times v (the inverse rotation)
pub fn apply_t(m: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| m[0][i] * v[0] + m[1][i] * v[1] + m[2][i] * v[2])
}

/// The picture key for a head turned (yaw, pitch) degrees: rounded to STEP, clamped.
pub fn key(yaw_deg: f64, pitch_deg: f64) -> String {
    let q = |v: f64, m: f64| ((v.clamp(-m, m) / STEP).round() * STEP) as i64;
    format!("{PREFIX}{}:{}", q(yaw_deg, MAX_YAW), q(pitch_deg, MAX_PITCH))
}

/// (yaw, pitch) in degrees from an orb key; None for other keys.
pub fn parse(key: &str) -> Option<(f64, f64)> {
    let (y, p) = key.strip_prefix(PREFIX)?.split_once(':')?;
    Some((y.parse().ok()?, p.parse().ok()?))
}

pub fn is_orb(key: &str) -> bool {
    key.starts_with(PREFIX)
}

fn norm(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn mix(a: [f64; 3], b: [f64; 3], k: f64) -> [f64; 3] {
    [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * k)
}

const YELLOW: [f64; 3] = [255.0, 203.0, 76.0];
const EAR: [f64; 3] = [240.0, 178.0, 60.0];
const EAR_IN: [f64; 3] = [205.0, 120.0, 55.0];
const DARK: [f64; 3] = [62.0, 40.0, 30.0];
const NOSE_C: [f64; 3] = [235.0, 120.0, 60.0];
const MOUTH: [f64; 3] = [150.0, 45.0, 45.0];
const RIM: [f64; 3] = [90.0, 55.0, 20.0];
/// wrapped Lambert for a light from the camera: (n.z + WRAP) / (1 + WRAP)
const WRAP: f64 = 0.5;

/// Colour (sRGB, 0..255) of the head sphere at head-space direction h (unit vector from the head centre).
pub fn head_colour(h: [f64; 3]) -> [f64; 3] {
    let eye_l = norm(MODEL[1]);
    let eye_r = norm(MODEL[2]);
    let nose = norm(MODEL[0]);
    if dot(h, eye_l) > 0.16f64.cos() || dot(h, eye_r) > 0.16f64.cos() {
        // a white glint up and to the image left of each pupil, fixed to the face
        let gl = norm([eye_l[0] - 0.07, eye_l[1] - 0.07, eye_l[2]]);
        let gr = norm([eye_r[0] - 0.07, eye_r[1] - 0.07, eye_r[2]]);
        if dot(h, gl) > 0.045f64.cos() || dot(h, gr) > 0.045f64.cos() {
            return [255.0, 255.0, 255.0];
        }
        return DARK;
    }
    if dot(h, nose) > 0.11f64.cos() {
        return NOSE_C;
    }
    if h[2] > 0.0 {
        // smile: an arc of elevation in azimuth, corners up
        let az = h[0].atan2(h[2]);
        let el = h[1].asin();
        let mid = 0.50 - 1.1 * az * az;
        if az.abs() < 0.42 && el > mid - 0.05 && el < mid + 0.05 {
            return MOUTH;
        }
    }
    YELLOW
}

/// The surface of an orb turned by r, seen at camera-space point (x, y) in model units (interocular distances,
/// head centre at the origin): (colour sRGB 0..255, camera-space normal) of the
/// nearest of the head and ear spheres, or None.
pub fn surface(r: &Mat3, x: f64, y: f64) -> Option<([f64; 3], [f64; 3])> {
    let ears = [[EAR_AT, MODEL[3][1], MODEL[3][2]], [-EAR_AT, MODEL[4][1], MODEL[4][2]]];
    let mut best: Option<(f64, [f64; 3], i32, f64)> = None; // (z, normal, which, rim)
    let spheres = [([0.0; 3], HEAD_R, -1), (ears[0], EAR_R, 0), (ears[1], EAR_R, 1)];
    for (c, rad, k) in spheres {
        let c = apply(r, c);
        let (dx, dy) = (x - c[0], y - c[1]);
        let d2 = dx * dx + dy * dy;
        if d2 >= rad * rad {
            continue;
        }
        let dz = (rad * rad - d2).sqrt();
        let z = c[2] + dz;
        if best.map_or(true, |b| z > b.0) {
            best = Some((z, [dx / rad, dy / rad, dz / rad], k, d2.sqrt() / rad));
        }
    }
    let (_, n, k, rim) = best?;
    let h = apply_t(r, n); // the surface normal in head space
    let mut base = if k < 0 {
        head_colour(h)
    } else {
        // inner ear: the part of the ear sphere facing sideways out of the head
        let side = if k == 0 { 1.0 } else { -1.0 };
        if h[0] * side > 0.55 && h[2] > -0.2 { mix(EAR, EAR_IN, 0.7) } else { EAR }
    };
    if rim > 0.93 {
        base = mix(base, RIM, 0.6);
    }
    Some((base, n))
}

/// The orb turned (yaw, pitch) degrees as a straight-alpha RGBA picture, size x size: the head sphere's diameter is
/// size / HALF, centred; shaded for a light from the camera; 4x4 supersampled.
pub fn picture(yaw_deg: f64, pitch_deg: f64, size: u32) -> Vec<u8> {
    let r = rot(yaw_deg.to_radians(), pitch_deg.to_radians(), 0.0);
    let unit = size as f64 / (2.0 * HALF * HEAD_R); // pixels per model unit
    let lin = |v: f64| {
        let x = v / 255.0;
        if x <= 0.04045 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) }
    };
    let srgb = |x: f64| {
        let x = x.clamp(0.0, 1.0);
        255.0 * if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
    };
    const SS: u32 = 4;
    let mut out = vec![0u8; (size * size * 4) as usize];
    for py in 0..size {
        for px in 0..size {
            let (mut acc, mut cover) = ([0.0f64; 3], 0.0);
            for sy in 0..SS {
                for sx in 0..SS {
                    let x = (px as f64 + (sx as f64 + 0.5) / SS as f64 - size as f64 / 2.0) / unit;
                    let y = (py as f64 + (sy as f64 + 0.5) / SS as f64 - size as f64 / 2.0) / unit;
                    let Some((c, n)) = surface(&r, x, y) else { continue };
                    let shade = ((n[2] + WRAP) / (1.0 + WRAP)).max(0.0);
                    for i in 0..3 {
                        acc[i] += lin(c[i]) * shade; // average in linear light
                    }
                    cover += 1.0;
                }
            }
            if cover == 0.0 {
                continue;
            }
            let i = ((py * size + px) * 4) as usize;
            for k in 0..3 {
                out[i + k] = srgb(acc[k] / cover).round() as u8;
            }
            out[i + 3] = (255.0 * cover / (SS * SS) as f64).round() as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_round_trip() {
        assert_eq!(key(13.0, -6.0), "orb:10:-10");
        assert_eq!(key(200.0, 99.0), "orb:90:40");
        assert_eq!(parse("orb:-30:20"), Some((-30.0, 20.0)));
        assert_eq!(parse("😀"), None);
    }

    #[test]
    fn picture_fits_its_square() {
        // however the head turns, nothing touches the picture's edge (the ears fit)
        for (y, p) in [(0.0, 0.0), (90.0, 0.0), (-90.0, 40.0), (45.0, -40.0)] {
            let n = 64u32;
            let img = picture(y, p, n);
            for i in 0..n {
                for (x, yy) in [(i, 0), (i, n - 1), (0, i), (n - 1, i)] {
                    assert_eq!(img[((yy * n + x) * 4 + 3) as usize], 0, "edge pixel at yaw {y} pitch {p}");
                }
            }
        }
    }
}
