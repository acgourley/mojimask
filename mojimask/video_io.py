"""One decoder (and encoder) for the whole app, so every step agrees on what "frame t" is.

Phones often record variable frame rate (VFR), especially in low light. Decoders disagree on such files
(one fills a constant-rate timeline, another returns only the stored frames); mixing them drew each
frame's covers on a different moment of the video. Here every step -- scan, camera motion, preview
frames, export -- reads frames from one constant-rate timeline at the video's nominal rate, upright
(rotation applied): frame t is the picture at time t / fps.

Backend: macOS's own AVFoundation (hardware decode/encode via VideoToolbox), no ffmpeg needed. On other
platforms (development only) it falls back to the ffmpeg command-line tools.

Colour: frames for the detectors ("detect") are converted from the decoder's YUV with OpenCV's BT.601
coefficients -- the detectors and tracker were tuned on such frames, and a BT.709 conversion cost ~5
points of held-face coverage on the test clip. Frames for the export ("display") come colour-managed from
AVFoundation, so the exported video's colours match the original.
"""
import json
import math
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from mojimask import about

try:  # macOS
    import AVFoundation as AV
    import CoreMedia as CM
    import objc
    import Quartz as Q
    from Foundation import NSURL
    HAVE_AVF = True
except ImportError:  # pragma: no cover
    HAVE_AVF = False

COMMON_RATES = (14.985, 15, 23.976, 24, 25, 29.97, 30, 47.952, 48, 50, 59.94, 60, 90, 100, 119.88, 120, 240)
TIMESCALE = 90000


def _snap(fps, tol=0.02):
    """Nearest common frame rate within tol, else None."""
    r = min(COMMON_RATES, key=lambda r: abs(fps - r))
    return float(r) if abs(fps - r) <= tol * r else None


# BT.601 limited-range YUV -> BGR exactly as OpenCV's ffmpeg-based reader produced the frames every
# detector threshold and tracker setting was tuned on: nearest-neighbour chroma, and coefficients fitted
# from decoded pixels (the decoded YUV is bit-identical between AVFoundation and ffmpeg; only this
# arithmetic differed -- swscale's fixed-point path runs ~1.5 levels darker than the textbook formula).
# Matches OpenCV to 0.26 levels on average (74% of pixels exactly); cv2.cvtColorTwoPlane was 1.25 off.
_YUV2BGR = np.array([[1.1642, 2.0167, 0.0, -278.2338],
                     [1.1645, -0.3897, -0.8121, 133.9585],
                     [1.1642, 0.0, 1.5956, -224.693]], np.float32)


def nv12_to_bgr(y, uv):
    """y: (H, W) uint8 luma; uv: (H/2, W/2, 2) uint8 interleaved chroma -> (H, W, 3) uint8 BGR."""
    H, W = y.shape
    uvn = cv2.resize(np.ascontiguousarray(uv), (W, H), interpolation=cv2.INTER_NEAREST)
    f = cv2.transform(np.dstack([y, uvn]).astype(np.float32), _YUV2BGR)
    return np.clip(np.rint(f), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ AVFoundation backend
def _asset(video):
    url = NSURL.fileURLWithPath_(os.path.abspath(video))
    return AV.AVURLAsset.URLAssetWithURL_options_(url, {AV.AVURLAssetPreferPreciseDurationAndTimingKey: True})


def _video_track(asset):
    tracks = asset.tracksWithMediaType_(AV.AVMediaTypeVideo)
    if not tracks:
        raise ValueError("no video track")
    return tracks[0]


def _rotation(track):
    t = track.preferredTransform()
    return int(round(math.degrees(math.atan2(t.b, t.a)))) % 360  # 0, 90, 180, 270 (clockwise)


def _upright(img, rot):
    k = {0: 0, 90: -1, 180: 2, 270: 1}.get(rot, 0)  # np.rot90: k=-1 turns clockwise
    return np.ascontiguousarray(np.rot90(img, k)) if k else img


def _typical_rate(asset, vt):
    """1 / the median gap between stored frames (timestamps only, nothing decoded), or None."""
    reader, _ = AV.AVAssetReader.alloc().initWithAsset_error_(asset, None)
    if reader is None:
        return None
    out = AV.AVAssetReaderTrackOutput.alloc().initWithTrack_outputSettings_(vt, None)
    out.setAlwaysCopiesSampleData_(False)
    reader.addOutput_(out)
    reader.startReading()
    ts = []
    try:
        while True:
            with objc.autorelease_pool():
                sb = out.copyNextSampleBuffer()
                if sb is None:
                    break
                if CM.CMSampleBufferGetNumSamples(sb):
                    ts.append(CM.CMTimeGetSeconds(CM.CMSampleBufferGetPresentationTimeStamp(sb)))
    finally:
        reader.cancelReading()
    gaps = np.diff(np.sort(ts))
    gaps = gaps[gaps > 0]
    return float(1 / np.median(gaps)) if len(gaps) else None


def _probe_avf(video):
    asset = _asset(video)
    vt = _video_track(asset)
    size = vt.naturalSize()
    w, h = int(round(size.width)), int(round(size.height))
    rot = _rotation(vt)
    if rot in (90, 270):
        w, h = h, w
    mfd = CM.CMTimeGetSeconds(vt.minFrameDuration())
    nominal = float(vt.nominalFrameRate())
    # Timeline rate: the shortest frame duration when the file states it; else the average rate when it is a
    # standard one; else (variable-rate phone video, e.g. 24 fps with 20 fps stretches in low light, whose
    # average is a non-standard 22.7) the typical gap between stored frames, snapped to a standard rate.
    fps = ((_snap(1 / mfd) if mfd and mfd > 0 else None) or _snap(nominal)
           or _snap(_typical_rate(asset, vt) or 0) or nominal or 30.0)
    tr = vt.timeRange()
    dur = CM.CMTimeGetSeconds(tr.duration)
    return {"width": w, "height": h, "fps": float(fps), "duration": float(CM.CMTimeGetSeconds(asset.duration())),
            "video_start": float(CM.CMTimeGetSeconds(tr.start)), "video_duration": float(dur), "rotation": rot,
            "has_audio": len(asset.tracksWithMediaType_(AV.AVMediaTypeAudio)) > 0}


def _plane(pb, i, rows, cols):
    base = Q.CVPixelBufferGetBaseAddressOfPlane(pb, i)
    rb = Q.CVPixelBufferGetBytesPerRowOfPlane(pb, i)
    return np.frombuffer(base.as_buffer(rb * rows), np.uint8).reshape(rows, rb)[:, :cols]


def _decoded(video, color):
    """(seconds, BGR image in stored orientation) for every stored frame, in presentation order."""
    asset = _asset(video)
    vt = _video_track(asset)
    reader, err = AV.AVAssetReader.alloc().initWithAsset_error_(asset, None)
    if reader is None:
        raise RuntimeError(f"can't read video: {err}")
    fmt = Q.kCVPixelFormatType_32BGRA if color == "display" else Q.kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
    out = AV.AVAssetReaderTrackOutput.alloc().initWithTrack_outputSettings_(vt, {Q.kCVPixelBufferPixelFormatTypeKey: fmt})
    out.setAlwaysCopiesSampleData_(False)
    reader.addOutput_(out)
    reader.startReading()
    try:
        while True:
            with objc.autorelease_pool():
                sb = out.copyNextSampleBuffer()
                if sb is None:
                    break
                pb = CM.CMSampleBufferGetImageBuffer(sb)
                if pb is None:
                    continue
                pts = CM.CMTimeGetSeconds(CM.CMSampleBufferGetPresentationTimeStamp(sb))
                Q.CVPixelBufferLockBaseAddress(pb, Q.kCVPixelBufferLock_ReadOnly)
                try:
                    W, H = Q.CVPixelBufferGetWidth(pb), Q.CVPixelBufferGetHeight(pb)
                    if fmt == Q.kCVPixelFormatType_32BGRA:
                        rb = Q.CVPixelBufferGetBytesPerRow(pb)
                        base = Q.CVPixelBufferGetBaseAddress(pb)
                        bgra = np.frombuffer(base.as_buffer(rb * H), np.uint8).reshape(H, rb)[:, :W * 4]
                        img = bgra.reshape(H, W, 4)[:, :, :3].copy()
                    else:  # NV12 -> BGR the way the detectors were tuned (see nv12_to_bgr)
                        y = _plane(pb, 0, H, W)
                        uv = _plane(pb, 1, H // 2, (W // 2) * 2).reshape(H // 2, W // 2, 2)
                        img = nv12_to_bgr(y, uv)
                finally:
                    Q.CVPixelBufferUnlockBaseAddress(pb, Q.kCVPixelBufferLock_ReadOnly)
            yield pts, img
        if reader.status() == AV.AVAssetReaderStatusFailed:
            raise RuntimeError(f"decode failed: {reader.error()}")
    finally:
        reader.cancelReading()


def _frames_avf(video, meta, color):
    """Constant-rate timeline: slot t shows the latest stored frame whose time rounds to <= t.
    Frames come out at the video's display size: a file whose pixels aren't square (anamorphic, e.g. an old
    film rip stored 708 wide to be shown 643 wide) decodes at its stored width, so each frame is resized to
    the shape it's meant to be seen at -- otherwise the boxes found on it and the picture every screen and
    the export draw (at the display size, probe's width) disagree, and the right edge goes missing."""
    fps, rot = meta["fps"], meta.get("rotation", 0)
    dw, dh = (meta["height"], meta["width"]) if rot in (90, 270) else (meta["width"], meta["height"])  # stored orientation
    t0 = meta.get("video_start", 0.0)
    n = max(1, round(meta.get("video_duration", meta["duration"]) * fps))
    src = _decoded(video, color)
    try:
        pending = next(src, None)
        cur = None
        for t in range(n):
            while pending is not None and (pending[0] - t0) * fps < t + 0.5:
                cur, pending = pending, next(src, None)
            if cur is None:
                if pending is None:
                    return
                cur = pending  # pad the start with the first frame
            img = cur[1]
            if img.shape[1] != dw or img.shape[0] != dh:
                img = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)
            yield _upright(img, rot)
    finally:
        src.close()


class _AVFWriter:
    """H.264 (or HEVC) .mp4 through AVAssetWriter; frames are BGR uint8 arrays of the given size."""

    def __init__(self, path, width, height, fps, bitrate, codec="h264", keyframes=None):
        if os.path.exists(path):
            os.remove(path)
        self.W, self.H, self.fps, self.n = width, height, fps, 0
        self.w, err = AV.AVAssetWriter.alloc().initWithURL_fileType_error_(NSURL.fileURLWithPath_(path),
                                                                           AV.AVFileTypeMPEG4, None)
        if self.w is None:
            raise RuntimeError(f"can't write {path}: {err}")
        props = {AV.AVVideoAverageBitRateKey: int(bitrate), AV.AVVideoMaxKeyFrameIntervalKey: int(keyframes or round(fps * 2)),
                 AV.AVVideoExpectedSourceFrameRateKey: int(round(fps))}
        if codec == "h264":
            props[AV.AVVideoProfileLevelKey] = AV.AVVideoProfileLevelH264HighAutoLevel
        settings = {AV.AVVideoCodecKey: AV.AVVideoCodecTypeHEVC if codec == "hevc" else AV.AVVideoCodecTypeH264,
                    AV.AVVideoWidthKey: width, AV.AVVideoHeightKey: height,
                    AV.AVVideoCompressionPropertiesKey: props,
                    AV.AVVideoColorPropertiesKey: {AV.AVVideoColorPrimariesKey: AV.AVVideoColorPrimaries_ITU_R_709_2,
                                                   AV.AVVideoTransferFunctionKey: AV.AVVideoTransferFunction_ITU_R_709_2,
                                                   AV.AVVideoYCbCrMatrixKey: AV.AVVideoYCbCrMatrix_ITU_R_709_2}}
        self.inp = AV.AVAssetWriterInput.assetWriterInputWithMediaType_outputSettings_(AV.AVMediaTypeVideo, settings)
        self.inp.setExpectsMediaDataInRealTime_(False)
        self.ad = AV.AVAssetWriterInputPixelBufferAdaptor.assetWriterInputPixelBufferAdaptorWithAssetWriterInput_sourcePixelBufferAttributes_(
            self.inp, {Q.kCVPixelBufferPixelFormatTypeKey: Q.kCVPixelFormatType_32BGRA,
                       Q.kCVPixelBufferWidthKey: width, Q.kCVPixelBufferHeightKey: height})
        self.w.addInput_(self.inp)
        if not self.w.startWriting():
            raise RuntimeError(f"can't start writing: {self.w.error()}")
        self.w.startSessionAtSourceTime_(CM.CMTimeMake(0, TIMESCALE))
        self.bgra = np.empty((height, width, 4), np.uint8)
        self.bgra[:, :, 3] = 255

    def write(self, bgr):
        while not self.inp.isReadyForMoreMediaData():
            if self.w.status() != AV.AVAssetWriterStatusWriting:  # failed or cancelled: it will never be ready
                raise RuntimeError(f"encode failed: {self.w.error()}")
            time.sleep(0.002)
        with objc.autorelease_pool():
            err, pb = Q.CVPixelBufferPoolCreatePixelBuffer(None, self.ad.pixelBufferPool(), None)
            if pb is None:
                raise RuntimeError(f"no pixel buffer ({err})")
            Q.CVPixelBufferLockBaseAddress(pb, 0)
            try:
                rb = Q.CVPixelBufferGetBytesPerRow(pb)
                dst = np.frombuffer(Q.CVPixelBufferGetBaseAddress(pb).as_buffer(rb * self.H), np.uint8).reshape(self.H, rb)
                self.bgra[:, :, :3] = bgr
                dst[:, :self.W * 4] = self.bgra.reshape(self.H, self.W * 4)
            finally:
                Q.CVPixelBufferUnlockBaseAddress(pb, 0)
            when = CM.CMTimeMake(int(round(self.n * TIMESCALE / self.fps)), TIMESCALE)
            if not self.ad.appendPixelBuffer_withPresentationTime_(pb, when):
                raise RuntimeError(f"encode failed: {self.w.error()}")
        self.n += 1

    def close(self):
        self.inp.markAsFinished()
        done = threading.Event()
        self.w.finishWritingWithCompletionHandler_(done.set)
        done.wait()
        if self.w.status() != AV.AVAssetWriterStatusCompleted:
            raise RuntimeError(f"encode failed: {self.w.error()}")


def _wait_export(session):
    done = threading.Event()
    session.exportAsynchronouslyWithCompletionHandler_(done.set)
    done.wait()
    return session.status() == AV.AVAssetExportSessionStatusCompleted, session.error()


def _mux_avf(video_only, audio_src, out, start=0.0, duration=None):
    """video_only's picture + audio_src's sound from `start` for the same duration -> out (.mp4).
    Built as a fresh composition, so no metadata (date, location, device) is carried over: the only metadata written
    is which app made it (about.NAME as the software, about.MADE_WITH as the description)."""
    v_asset = _asset(video_only)
    v_src = _video_track(v_asset)
    comp = AV.AVMutableComposition.composition()
    v_dur = v_asset.duration()
    if duration is not None:
        v_dur = CM.CMTimeMinimum(v_dur, CM.CMTimeMakeWithSeconds(duration, TIMESCALE))
    vt = comp.addMutableTrackWithMediaType_preferredTrackID_(AV.AVMediaTypeVideo, CM.kCMPersistentTrackID_Invalid)
    ok, err = vt.insertTimeRange_ofTrack_atTime_error_(CM.CMTimeRangeMake(CM.kCMTimeZero, v_dur), v_src, CM.kCMTimeZero, None)
    if not ok:
        raise RuntimeError(f"mux failed: {err}")
    a_asset = _asset(audio_src) if audio_src else None
    a_tracks = a_asset.tracksWithMediaType_(AV.AVMediaTypeAudio) if a_asset else []
    if a_tracks:
        a_range = CM.CMTimeRangeMake(CM.CMTimeMakeWithSeconds(start, TIMESCALE), v_dur)
        at = comp.addMutableTrackWithMediaType_preferredTrackID_(AV.AVMediaTypeAudio, CM.kCMPersistentTrackID_Invalid)
        ok, err = at.insertTimeRange_ofTrack_atTime_error_(a_range, a_tracks[0], CM.kCMTimeZero, None)
        if not ok:  # the video still exports, silent
            comp.removeTrack_(at)
            print(f"video_io: no sound in the export, couldn't add it: {err}", file=sys.stderr, flush=True)
    if os.path.exists(out):
        os.remove(out)
    for preset in (AV.AVAssetExportPresetPassthrough, AV.AVAssetExportPresetHighestQuality):
        s = AV.AVAssetExportSession.alloc().initWithAsset_presetName_(comp, preset)
        s.setOutputURL_(NSURL.fileURLWithPath_(out))
        s.setOutputFileType_(AV.AVFileTypeMPEG4)
        s.setShouldOptimizeForNetworkUse_(True)
        s.setMetadata_(_made_by_avf())
        ok, err = _wait_export(s)
        if ok:
            return
        if os.path.exists(out):
            os.remove(out)
    raise RuntimeError(f"mux failed: {err}")


def _made_by_avf():
    """The common keys (an .mp4 keeps these; the iTunes and QuickTime user-data ones are dropped from it)."""
    items = []
    for key, value in ((AV.AVMetadataCommonKeySoftware, about.NAME), (AV.AVMetadataCommonKeyDescription, about.MADE_WITH)):
        it = AV.AVMutableMetadataItem.metadataItem()
        it.setKeySpace_(AV.AVMetadataKeySpaceCommon)
        it.setKey_(key)
        it.setValue_(value)
        items.append(it)
    return items


# ------------------------------------------------------------------ ffmpeg fallback (development off macOS)
def _probe_ffmpeg(video):
    """Best effort (development off macOS): the same keys as _probe_avf, but no pixel-aspect correction."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,avg_frame_rate:stream_side_data=rotation:format=duration",
                          "-of", "json", video], capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    st = d["streams"][0]
    rot = 0
    for sd in st.get("side_data_list", []) or []:
        if "rotation" in sd:  # (ffprobe also lists empty side-data entries; don't let them reset it)
            rot = int(sd["rotation"] or 0)
    w, h = st["width"], st["height"]
    if abs(rot) in (90, 270):
        w, h = h, w

    def rate(s):
        num, den = map(int, (s or "0/0").split("/"))
        return num / den if den else 0.0
    fps = rate(st.get("r_frame_rate"))
    if not 1 <= fps <= 240:
        fps = rate(st.get("avg_frame_rate")) or 30.0
    dur = float(d["format"]["duration"])
    audio = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                            "-of", "csv=p=0", video], capture_output=True, text=True).stdout.strip()
    return {"width": w, "height": h, "fps": fps, "duration": dur, "video_start": 0.0, "video_duration": dur,
            "rotation": (-rot) % 360,  # ffprobe's is counter-clockwise; ffmpeg's frames come out upright
            "has_audio": bool(audio)}


def _timeline(meta, color="detect"):
    """ffmpeg output options for the shared constant-rate timeline (BT.601 colours for the detectors)."""
    vf = ["-vf", "scale=in_color_matrix=bt601"] if color == "detect" else []
    return ["-fps_mode", "cfr", "-r", f"{meta['fps']:.6f}", *vf]


def _frames_ffmpeg(video, meta, color):
    W, H = meta["width"], meta["height"]
    size = W * H * 3
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", video, *_timeline(meta, color),
                            "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"], stdout=subprocess.PIPE)
    try:
        while True:
            buf = dec.stdout.read(size)
            if len(buf) < size:
                return
            yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    finally:
        dec.stdout.close()
        dec.kill()
        dec.wait()


class _FFmpegWriter:
    def __init__(self, path, width, height, fps, bitrate, codec="h264", keyframes=None):
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                   "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
                                   "-c:v", "libx264", "-b:v", str(int(bitrate)), "-pix_fmt", "yuv420p",
                                   "-g", str(int(keyframes or round(fps * 2))), path],
                                  stdin=subprocess.PIPE)

    def write(self, bgr):
        self.p.stdin.write(np.ascontiguousarray(bgr).tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError("ffmpeg encode failed")


def _mux_ffmpeg(video_only, audio_src, out, start=0.0, duration=None):
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", video_only]
    if audio_src:
        cmd += ["-ss", f"{start:.3f}", "-i", audio_src, "-map", "0:v", "-map", "1:a?"]
    cmd += ["-map_metadata", "-1", "-metadata", f"encoder={about.NAME}", "-metadata", f"comment={about.MADE_WITH}", "-c", "copy", "-shortest", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)


# ------------------------------------------------------------------ photos
# A photo is a clip of one frame: the scan, the questions, the editor and the export all work on frame 0, and
# the export writes a photo back (track.render). Read by macOS's ImageIO (HEIC from iPhones, JPEG, PNG, ...),
# turned upright by its orientation tag and colour-managed into sRGB, the space the detectors were tuned in.
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".webp", ".bmp")
PHOTO_FPS = 30.0  # the one frame's nominal rate (the timeline needs one; nothing moves)


def is_image(path):
    return str(path).lower().endswith(IMAGE_EXT)


def _read_image_cg(path):
    src = Q.CGImageSourceCreateWithURL(NSURL.fileURLWithPath_(os.path.abspath(path)), None)
    if src is None or not Q.CGImageSourceGetCount(src):
        raise ValueError("not a picture macOS can read")
    props = Q.CGImageSourceCopyPropertiesAtIndex(src, 0, None) or {}
    side = max(int(props.get(Q.kCGImagePropertyPixelWidth, 0)), int(props.get(Q.kCGImagePropertyPixelHeight, 0)))
    opts = {Q.kCGImageSourceCreateThumbnailFromImageAlways: True,  # (the full-size image, with the orientation tag applied)
            Q.kCGImageSourceCreateThumbnailWithTransform: True, Q.kCGImageSourceShouldCacheImmediately: True}
    if side:
        opts[Q.kCGImageSourceThumbnailMaxPixelSize] = side
    img = Q.CGImageSourceCreateThumbnailAtIndex(src, 0, opts)
    if img is None:
        raise ValueError("not a picture macOS can read")
    W, H = Q.CGImageGetWidth(img), Q.CGImageGetHeight(img)
    buf = bytearray(W * H * 4)
    ctx = Q.CGBitmapContextCreate(buf, W, H, 8, W * 4, Q.CGColorSpaceCreateWithName(Q.kCGColorSpaceSRGB),
                                  Q.kCGImageAlphaNoneSkipLast)
    Q.CGContextSetRGBFillColor(ctx, 1, 1, 1, 1)  # transparent parts on white
    Q.CGContextFillRect(ctx, Q.CGRectMake(0, 0, W, H))
    Q.CGContextDrawImage(ctx, Q.CGRectMake(0, 0, W, H), img)
    del ctx
    return np.ascontiguousarray(np.frombuffer(buf, np.uint8).reshape(H, W, 4)[:, :, 2::-1])  # RGBX -> BGR


def read_image(path):
    """A photo as an upright BGR uint8 array."""
    if HAVE_AVF:
        return _read_image_cg(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # (applies the EXIF orientation)
    if img is None:
        raise ValueError("not a picture OpenCV can read")
    return img


def _probe_image(path):
    h, w = read_image(path).shape[:2]
    d = 1 / PHOTO_FPS
    return {"width": int(w), "height": int(h), "fps": PHOTO_FPS, "duration": d, "video_start": 0.0, "video_duration": d,
            "rotation": 0, "has_audio": False, "image": True}


def write_image(path, bgr, quality=92):
    """A photo with none of the original's metadata (no date, place or camera), only which app made it (EXIF
    Software and ImageDescription; for PNG, text chunks): JPEG, or PNG for a .png name."""
    from PIL import Image, PngImagePlugin
    img = Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))
    if os.path.splitext(path)[1].lower() == ".png":
        info = PngImagePlugin.PngInfo()
        info.add_text("Software", about.NAME)
        info.add_text("Comment", about.MADE_WITH)
        img.save(path, "PNG", pnginfo=info, compress_level=6)
    else:
        exif = Image.Exif()
        exif[0x0131] = about.NAME  # Software
        exif[0x010E] = about.MADE_WITH  # ImageDescription
        img.save(path, "JPEG", quality=quality, exif=exif)


# ------------------------------------------------------------------ public API
SPACE_VERSION = 2  # how frames are decoded (coords.py): 1 = stored size, 2 = display size, square pixels


def space(meta):
    """The pixel space this decoder's frames are in, for a video's probe() info: what every saved position of
    a project is measured in (coords.py)."""
    return {"w": int(meta["width"]), "h": int(meta["height"]), "rot": int(meta.get("rotation") or 0), "v": SPACE_VERSION}


def probe(video, keep=None):
    """Upright size, timeline fps, duration (+ rotation, has_audio on macOS).
    keep: a project's saved meta -- its timeline rate wins, so a project scanned before a change to the rate
    rule keeps the frame numbering its frames, detections and edits were made on."""
    if is_image(video):
        return _probe_image(video)
    meta = _probe_avf(video) if HAVE_AVF else _probe_ffmpeg(video)
    if keep and keep.get("fps"):
        meta["fps"] = float(keep["fps"])
    return meta


def frames(video, meta=None, color="detect"):
    """Yield upright BGR uint8 frames (H, W, 3) on the shared timeline. color: "detect" | "display"."""
    if is_image(video):
        return (f for f in [read_image(video)])  # (a generator, like the video decoders: callers .close() it)
    meta = meta or probe(video)
    return _frames_avf(video, meta, color) if HAVE_AVF else _frames_ffmpeg(video, meta, color)


def extract_jpegs(video, out_dir, meta=None, quality=90):
    """Write out_dir/00000.jpg, ... on the shared timeline (the review screens show these)."""
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for n, f in enumerate(frames(video, meta), 1):
        cv2.imwrite(f"{out_dir}/{n - 1:05d}.jpg", f, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return n


def writer(path, width, height, fps, bitrate, codec="h264", keyframes=None):
    """Video-only encoder: .write(bgr) per frame, then .close(). keyframes: at most this many frames apart
    (default 2 s; depth.py's grey depth movies use fewer, to seek quickly)."""
    return (_AVFWriter if HAVE_AVF else _FFmpegWriter)(path, width, height, fps, bitrate, codec, keyframes)


def luma(video, start=0.0, end=None):
    """(seconds, Y plane as (H, W) uint8) for the stored frames from `start` (seconds) to `end`, decoding from the
    keyframe before start. The raw luma, no colour conversion: for grey data movies (depth.py), not pictures."""
    if not HAVE_AVF:
        cap = cv2.VideoCapture(video)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start * fps)))
            i = int(round(start * fps))
            while True:
                ok, f = cap.read()
                if not ok or (end is not None and i / fps > end + 1e-6):
                    break
                yield i / fps, cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)  # (FFmpeg path: grey through BGR; depth.py calibrates)
                i += 1
        finally:
            cap.release()
        return
    asset = _asset(video)
    vt = _video_track(asset)
    reader, err = AV.AVAssetReader.alloc().initWithAsset_error_(asset, None)
    if reader is None:
        raise RuntimeError(f"can't read {video}: {err}")
    if start > 0 or end is not None:
        dur = CM.kCMTimePositiveInfinity if end is None else CM.CMTimeMakeWithSeconds(end - start + 1e-3, TIMESCALE)
        reader.setTimeRange_(CM.CMTimeRangeMake(CM.CMTimeMakeWithSeconds(start, TIMESCALE), dur))
    fmt = Q.kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
    out = AV.AVAssetReaderTrackOutput.alloc().initWithTrack_outputSettings_(vt, {Q.kCVPixelBufferPixelFormatTypeKey: fmt})
    out.setAlwaysCopiesSampleData_(False)
    reader.addOutput_(out)
    reader.startReading()
    try:
        while True:
            with objc.autorelease_pool():
                sb = out.copyNextSampleBuffer()
                if sb is None:
                    break
                pb = CM.CMSampleBufferGetImageBuffer(sb)
                if pb is None:
                    continue
                pts = CM.CMTimeGetSeconds(CM.CMSampleBufferGetPresentationTimeStamp(sb))
                Q.CVPixelBufferLockBaseAddress(pb, Q.kCVPixelBufferLock_ReadOnly)
                try:
                    y = _plane(pb, 0, Q.CVPixelBufferGetHeight(pb), Q.CVPixelBufferGetWidth(pb)).copy()
                finally:
                    Q.CVPixelBufferUnlockBaseAddress(pb, Q.kCVPixelBufferLock_ReadOnly)
            yield pts, y
        if reader.status() == AV.AVAssetReaderStatusFailed:
            raise RuntimeError(f"decode failed: {reader.error()}")
    finally:
        reader.cancelReading()


def mux(video_only, audio_src, out, start=0.0, duration=None):
    """Add audio_src's sound (from `start`) to video_only -> out. No metadata is copied."""
    return (_mux_avf if HAVE_AVF else _mux_ffmpeg)(video_only, audio_src, out, start, duration)
