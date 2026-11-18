//! mm-render: everything drawn on the GPU, through wgpu.
//!
//! - [`device`]: a wgpu device of our own (headless: export, CLI) or adopted from the host's (Qt's MTLDevice and
//!   queue, so wgpu submits onto Qt's own queue), and wrapping the host's texture.
//! - [`yuv`]: decoded planes (NV12/P010) -> RGB at the stored size; [`prep`]: resize to square pixels + rotation.
//! - [`compositor`]: the WGSL port of `mojimask/gpu_draw.py`'s kernel (guide + compose) and its picture atlas.
//! - [`nv12`]: RGB -> NV12 bytes for encoders that can't take our pixel buffers; [`readback`]: textures -> bytes/PNG.
//! - [`sprites`]: the cover pictures (emoji through CoreText, portrait photos) and their atlas slices.
//! - [`styles`]: the blur, pixelate and solid covers; [`mark`]: the export's corner mark.
//! - [`draw`]: one call that draws a frame's records in any style (what the app and `mm-render-ref` use).
pub mod compositor;
pub mod device;
pub mod draw;
pub mod mark;
pub mod nv12;
pub mod prep;
pub mod readback;
pub mod sprites;
pub mod styles;
pub mod yuv;

pub use compositor::{out_texture, Compositor};
pub use device::Gpu;
