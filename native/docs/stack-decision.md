# Which front end draws mojimask's pictures

_The proofs of concept this compares (`webgpu/`, `rust/`, `qt/`, `ml/`, `harness/`, each with its REPORT.md) were
under `poc/`, a working folder that isn't in the published history: this page keeps what they found. Paths below
are relative to that `poc/`._

Question: if the Python engine only decides (cover records) and the client draws every final picture (preview and
export), which client stack is best, given that the app should go cross-platform later? Three proofs of concept
were run against the same reference (`harness/`, today's Metal output on jazz): `webgpu/`, `rust/`, `qt/`.
Their REPORT.md files have the details. All measured on an M2 Ultra, macOS 15.7, 2026-11-17. The speed numbers
are from re-running each `bench.sh` alone, after the PoCs had finished.

## Results side by side

| | WebGPU, Chrome (Electron) | WebGPU, WKWebView | Rust + wgpu | Qt (PySide6 + QRhi) |
|---|---|---|---|---|
| Compositor port, jazz min PSNR | 69.0 dB | 69.0 dB | 79.6 dB | 97.8 dB |
| Decode matches AVFoundation (jazz/4k/rot/vfr) | yes, 49-55 dB after 4 fixes | yes, 45-47 dB | yes, 75-78 dB (AVFoundation); FFmpeg one frame off on jazz | one frame off on jazz; anamorphic wrong |
| HDR HLG 10-bit | 12.7 dB (uncontrolled tone map) | 45.9 dB | 75.1 dB | 51.6 dB (own shader) |
| CPU pixel copies, decode to GPU | 0 | 0 | 0 | 1 (texture only via private API) |
| 4K decode+composite | 141 fps | 71 fps | 134 fps | 95 fps |
| 4K seek median / p95 | 39 / 60 ms | 59 / 80 ms | 84 / 108 ms | 202 / 283 ms |
| 4K export, 127 frames, AAC copied | 1.4 s, yes | 3.6 s, yes | 2.2 s, yes | 5.7 s, audio needs ffmpeg; fps bug |
| Editor UI | today's HTML/JS, unchanged | same | rewrite (egui, not Mac-like) | rewrite (Widgets, mostly native) |
| Edit-to-see | 0.4 s, no build | 0.65 s | ~2 s rebuild | ~1.8 s |
| App size (Mac) | not built (Electron est. 250 MB+) | 27 MB | 12 MB | 133 MB |
| Windows/Linux media | the browser's (Media Foundation; Linux weak) | WebView2 / WebKitGTK (no WebGPU) | ours to write per OS | ours to write (Qt Multimedia unfit) |

## Decision: WebGPU in a Chromium shell (Electron), Python engine beside it

- **It already has a media layer for every OS.** The hardest part of mojimask isn't drawing, which every stack ported
  within 1-2 levels. It's video in and out: frame-exact decode, edit lists, anamorphic pixels, colour, and an encoder
  that leaves the audio untouched. With Chromium, the browser supplies hardware decode and encode on each OS
  (VideoToolbox on Mac, Media Foundation on Windows). With Rust or Qt we would write that layer ourselves, per OS.
- **Speed is as good as native.** On the Mac it matches or beats Rust: playback 141 vs 134 fps, seek 39 vs 84 ms,
  export 1.4 vs 2.2 s.
- **The UI and how we work on it carry over:** `web/`, `theme.css`, `icons.js`, the pen.dev designs, and
  iterating in localhost.
- **Not WKWebView** (pywebview or Tauri on the Mac): WebGPU there needs a private switch on macOS 15, composes 3x
  slower, and has two bugs we had to work around. It also doesn't exist in WebKitGTK on Linux.
- **Rust + wgpu is the fallback.** It's the best engine (smallest app, most control, best HDR) but the most work:
  a new UI plus per-OS media code. Its compositor is WGSL like the WebGPU one, so the shader would carry over.
- **Qt is out.** QRhi is excellent for drawing, but Qt Multimedia is a player, not a frame-exact decoder or encoder.
  Fixing that means the same per-OS media work as Rust, plus a 133 MB Python app.

## What this choice still has to prove

1. **Electron itself.** Untested here, because the install was refused in the PoC session. Still to measure: size,
   start time, starting the Python engine as a child process, signing and notarization, and memory use.
2. **Colour depends on the engine.** Chrome treats BT.709 as Apple's gamma on import (undone in WGSL) and doesn't
   let us control HLG on the zero-copy path. Electron pins its Chromium version, which helps. The harness's parity
   test should run on every Electron upgrade. HDR needs the 10-bit-planes path (one copy per frame), not built.
3. **Linux.** Chromium's WebGPU there is a staged rollout and hardware decode is usually off. Expect Linux to
   come last and run on the CPU.
4. **Formats Chromium won't decode.** WebCodecs has no ProRes (iPhones record it; AVFoundation reads it). Those
   clips would need the engine to decode for the client. See `GYROFLOW.md`, whose native stack exists for exactly
   these formats.

## Second round: Rust + Qt Quick + GStreamer + ONNX Runtime (2026-11-17)

Afterwards the Rust PoC gained:
- a GStreamer media layer (`rust/gst/`);
- a Qt Quick shell, with wgpu drawing on Qt's own Metal device and queue (`rust/qtshell/`);
- the engine's ML models run from Rust through `ort` (`ml/`).

Each folder has a REPORT.md. What changed:

| Gap the first round found in native stacks | Result now |
|---|---|
| Per-OS media layer to write ourselves | GStreamer (LGPL): edit list right (jazz not shifted), all 5 clips 52-55 dB vs AVFoundation, 4K 121-129 fps, seek 71 ms, export 2 s, 0 CPU copies. One demuxer and muxer on every OS |
| UI rewrite, slow iteration | Qt Quick: QML live reload 55-83 ms; native menus, dialogs, drop and dark mode; frame shown matches ref (99 dB); scrub and drag equal to egui |
| Engine needs Python | Every model runs from Rust with outputs identical to Python on the same engine, and identical boxes; same speed on CoreML; DirectML/CUDA/WebGPU EPs prebuilt for Windows and Linux |

**Revised recommendation: Rust (wgpu + Qt Quick + GStreamer + ort) now leads; Electron + WebGPU is second.**
- **One decoder for engine and client.** With GStreamer in both, "frame t" and colour are defined once. With
  Electron, the client decodes with Chromium and the engine with something else, forever. That parity risk showed
  up in every round (Chrome's gamma quirk, FFmpeg's one-frame shift).
- **The engine can become cross-platform** through the same pieces (GStreamer, ort), without Python.
- **No Chromium.** Everything runs in one process, with no Chromium version churn.

What it costs:
- **Rewrite the UI in QML.** `web/` doesn't carry over, though QML's live reload keeps iteration fast.
- **Port the engine to Rust over time.** Keeping the Python engine beside the client is possible meanwhile.
- **Two Mac-specific risks.** Mac zero-copy relies on private GStreamer internals (the fallback is one CPU copy,
  still 87-94 fps at 4K). mp4mux drops the 2 AAC priming packets.
- **A heavy app.** Qt is 154 MB untrimmed.

Electron still wins if keeping today's HTML UI matters most. Neither stack has been built on Windows or Linux.

## Found along the way (true whichever stack is chosen)

- **"Frame t" must be defined once.** Decoders disagree: FFmpeg and Qt put jazz one frame off (an edit list that
  starts inside a frame), mediabunny gets its length wrong, and two of them handle anamorphic pixels differently.
  The engine (which scans with AVFoundation) should put each frame's source timestamp in the cover records, and the
  client should seek by timestamp, not by its own arithmetic.
- **The engine isn't cross-platform yet either.** `video_io` (AVFoundation) and the CoreML detector plans are
  Mac-only. On Windows the engine needs another decoder, and it must agree with the client's (see above).
- **Emoji:** Apple Color Emoji can't ship on Windows or Linux. Ship pre-drawn emoji pictures (e.g. Noto, OFL)
  everywhere, so no stack has to draw them from a font. That changes the look on the Mac too, so it's a product decision.
- **Some emoji work is drawing, not deciding:** covers tilt and turn with the head (`head_pose.posed_sprite`), which
  is a warp done on the CPU today: 684 distinct pictures for 127 jazz frames. In the new split that warp should move
  into the shader (sample through a transform), and the records should carry the pose instead of pictures.
- **Possible bug in today's app:** Python's HTTPServer looks up the hostname of 127.0.0.1, which hung for 35 s at
  start on this Mac. Worth checking `app_server` and `review_server` for the same thing.
