//! How a decoded YUV frame becomes RGB: the stream's matrix, range and chroma siting, as the media layer reads them
//! from the caps and the renderer's yuv pass applies them.
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct YuvParams {
    /// luma weights of R and B (BT.709: 0.2126, 0.0722)
    pub kr: f32,
    pub kb: f32,
    /// full range (0-255) rather than video range (16-235)
    pub full: bool,
    /// chroma co-sited with luma horizontally / vertically
    pub hcos: bool,
    pub vcos: bool,
    /// 8 or 10 bits (P010 carries 10 bits in the top of 16)
    pub depth: u32,
    /// HLG transfer: convert to SDR (false = show the signal through the matrix only, as AVFoundation does)
    pub hlg_to_sdr: bool,
    /// When set, the whole conversion as a 3x4 affine matrix on codes in 8-bit units (rows R, G, B; columns Y, Cb,
    /// Cr, 1; 10-bit codes divided by 4 first), rounded floor(x + 0.5): AVFoundation's 32BGRA conversion as
    /// mm_media::convert::display_matrix measures it. It replaces kr, kb and full (and HLG to SDR).
    #[serde(default)]
    pub matrix: Option<[[f32; 4]; 3]>,
}

impl Default for YuvParams {
    fn default() -> Self {
        YuvParams { kr: 0.2126, kb: 0.0722, full: false, hcos: false, vcos: false, depth: 8, hlg_to_sdr: false, matrix: None }
    }
}

// ------------------------------------------------------------------ lighting: applying an estimate (lighting.py)
//
// Scene-matched emoji lighting (lighting.py): each emoji is a matte cardboard cut-out standing where the face is. The
// AUTO estimate (per frame the scene's colour cast and exposure, per cover the light on that face and a direction) is
// computed by the engine; the USER settings (edits["light"]) are applied on top here, as `apply_settings` and
// `film_settings` do (and light_ui.js lightFor()/filmFor()). The shader then does, in LINEAR light (BGR):
//     albedo = desat(albedo, saturation)               # the Saturation slider (0 = gray, 1 = the emoji's own)
//     lit = albedo * (key * ramp(x, y) + ambient)
//     lit = max(lit, albedo * floor)                  # legibility floor
//     out = lerp(albedo, lit, mix)                     # mix 0 = flat emoji
//     out = black + out * (1 - black)                  # FILM: the footage's black level (old film is never black)
// key = level * key_auto, or with a Light colour picked: its brightness (mean channel) in that colour; ambient =
// amb * scene mean colour; ramp = 1 + ramp_strength * (gx * x + gy * y). Export also adds grain matched to the frame
// and motion blur along the cover's velocity.

/// lighting.WHEEL_SAT: the Light colour wheel's radius 1 is this saturated.
pub const WHEEL_SAT: f64 = 0.8;

/// The user's lighting settings (lighting.DEFAULTS; edits["light"] overrides).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct LightSettings {
    pub enabled: bool,
    pub level: f64,
    pub mix: f64,
    pub amb: f64,
    pub ramp: f64,
    pub floor: f64,
    /// None = the estimated light colour, or [hue_deg, radius 0..1] from the Light colour wheel
    pub light_colour: Option<[f64; 2]>,
    pub grain: bool,
    pub blur: bool,
    /// motion blur length as a fraction of one frame's movement
    pub shutter: f64,
    pub saturation: f64,
    /// multiplier on the footage's measured black level
    pub black: f64,
}

impl Default for LightSettings {
    fn default() -> Self {
        LightSettings { enabled: true, level: 1.0, mix: 1.0, amb: 0.15, ramp: 0.8, floor: 0.12, light_colour: None,
                        grain: true, blur: true, shutter: 1.0, saturation: 1.0, black: 1.0 }
    }
}

/// colorsys.hsv_to_rgb
pub fn hsv_to_rgb(h: f64, s: f64, v: f64) -> [f64; 3] {
    if s == 0.0 {
        return [v, v, v];
    }
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f)));
    match (i as i64).rem_euclid(6) {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

/// lighting.chroma: normalise so the mean channel is 1.
pub fn chroma(c: [f64; 3]) -> [f64; 3] {
    let m = ((c[0] + c[1] + c[2]) / 3.0).max(1e-6);
    [c[0] / m, c[1] / m, c[2] / m]
}

/// lighting.wheel_bgr: the Light colour wheel [hue_deg, radius] -> light colour normalised to mean 1, BGR.
pub fn wheel_bgr(wheel: [f64; 2]) -> [f64; 3] {
    let [r, g, b] = hsv_to_rgb(wheel[0].rem_euclid(360.0) / 360.0, wheel[1].clamp(0.0, 1.0) * WHEEL_SAT, 1.0);
    chroma([b, g, r])
}

/// lighting.apply_settings: auto estimate + user settings -> (key, ambient, ramp gradient), BGR.
pub fn apply_settings(e: &crate::cover::LightEstimate, st: &LightSettings) -> ([f64; 3], [f64; 3], [f64; 2]) {
    let mut key = e.key.map(|k| st.level * k);
    if let Some(w) = st.light_colour {
        let m = (key[0] + key[1] + key[2]) / 3.0;
        key = wheel_bgr(w).map(|c| m * c);
    }
    (key, e.amb.map(|a| st.amb * a), e.grad.map(|g| g * st.ramp))
}

/// lighting.film_settings: (saturation, black) from the user's saturation and the footage's measured black level times
/// the user's multiplier.
pub fn film_settings(black_auto: f64, st: &LightSettings) -> (f64, f64) {
    (st.saturation.clamp(0.0, 1.0), (black_auto * st.black).clamp(0.0, 0.5))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wheel_and_settings() {
        // colorsys.hsv_to_rgb(30/360, 0.48, 1) = (1, 0.76, 0.52) -> BGR normalised to mean 1
        let w = wheel_bgr([30.0, 0.6]);
        assert!((w[0] - 0.52 / (2.28 / 3.0)).abs() < 1e-9 && (w[2] - 1.0 / (2.28 / 3.0)).abs() < 1e-9);
        let st = LightSettings { saturation: 2.0, black: 3.0, ..Default::default() };
        assert_eq!(film_settings(0.2, &st), (1.0, 0.5));
    }
}
