//! Fails loudly if applemedia's private metas change: the zero-copy import reads GstCoreVideoMeta and the zero-copy
//! encode writes GstCoreMediaMeta as structs we declared from gst-plugins-bad's coremediabuffer.h. If this fails,
//! the app still works (one CPU copy per frame), but update mac/cv.rs.
#![cfg(target_os = "macos")]

fn jazz() -> String {
    format!("{}/../../../test_clips/jazz.mp4", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn core_video_meta_layout_unchanged() {
    let mut d = mm_media::Decoder::open(&jazz()).expect("open jazz");
    let f = d.frame_at(0).unwrap().unwrap();
    assert!(mm_media::mac::cv::zero_copy_available(f.buffer.as_ref()), "vtdec buffer has no usable GstCoreVideoMeta");
    if let Err(e) = mm_media::mac::cv::video_meta_ok() {
        panic!("GstCoreVideoMeta changed: {e}");
    }
}

#[test]
fn core_media_meta_layout_unchanged() {
    mm_media::decode::init();
    let pool = mm_media::mac::cv::Pool::new(64, 64, objc2_core_video::kCVPixelFormatType_32BGRA).unwrap();
    let pb = pool.get().unwrap();
    mm_media::mac::cv::wrap_pixbuf(&pb).expect("GstCoreMediaMeta changed");
}
