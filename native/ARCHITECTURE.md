# Mojimask native: architecture

This is the Rust rewrite of mojimask: a native app on Mac now, and Windows and Linux later. The stack comes from
`native/docs/stack-decision.md`:
- **wgpu** for drawing;
- **GStreamer** for video in and out;
- **ONNX Runtime** (`ort`) for the models;
- **Qt Quick** for the window.

The Python app (`mojimask/`, `web/`) stays the product until this one does everything it does.

## Three rules that shape everything

1. **One store, driven by commands.**
   - All app state lives in Rust as one serialisable tree.
   - QML reads it (`store.state.editor.t`) and changes it only through `store.call("seek", {t: 10})`.
   - The agent hooks call the same `call`, so an agent drives the app exactly as a click does (like
     `decisions_cli.py` today).
   - There's no business logic in QML or C++.
2. **The C++ is a thin host.** It does four things and nothing else:
   - shows QML;
   - hosts the video item, which wgpu draws into;
   - bridges the store;
   - serves the developer hooks.

   Everything else is Rust. The bridge is a small C ABI, and adding a feature never touches it: new commands and
   state fields flow through as JSON.
3. **One media layer for everything.** The engine's scan, the editor and the export all decode through `mm-media`.
   So "frame t" (a source timestamp) and colour are defined once (`native/docs/stack-decision.md`).

## Crates (`native/crates/`)

| crate | what it owns | depends on |
|---|---|---|
| `mm-core` | Plain data, rules and the project files (no GPU, no media): time (frame t ↔ source pts), geometry, cover records (`CoverRow`, the kernel's 36 floats), project model (facts log, edits), and later the scan's rules (blips, overlaps, same, cover) | – |
| `mm-media` | GStreamer: `probe`, `Decoder` (seek/next with source pts), `Exporter` (encode, mux, audio copied). Per-OS frame import into wgpu (`mac/`: CVPixelBuffer → Metal; later `win/` D3D12, `linux/` DMABuf/Vulkan) | mm-core |
| `mm-render` | wgpu: device (its own or adopted from Qt), the WGSL compositor (guide + compose), YUV→RGB, prep (resize/rotate), styles, sprite atlas, readback | mm-core |
| `mm-ml` | `ort`: sessions, plans and execution providers per platform, pre/post-processing (letterbox, blob, NMS, RetinaFace decode), the model registry | mm-core |
| `mm-vision` | Image functions, bit-exact with the project's cv2 (resize, warpAffine, GaussianBlur, cvtColor, erode, borders, polygon test) | – |
| `mm-motion` | Our own motion (the user's call): camera motion between frames and following a head through a gap. GPU first, judged by known-motion tests and the decisions benchmark, not by matching cv2 | core, vision, render |
| `mm-analysis` | Per-detection measurements: RetinaFace landmarks + SFace (reid.py), OSNet (appearance.py), depth (depth.py), pose helpers (pose_util.py, head_pose.py) | core, vision, ml |
| `mm-scan` | The scan's decide step (mojimask/scan/): tracklets, join/follow, faces, graph, attach, cover/behind, blips, overlaps, same, questions, newalgo. Each stage reads and writes the same proto/ files as Python | core, vision, analysis (flow and follow behind traits; mm-motion supplies them) |
| `mm-engine` | The work: detection on worker threads, running mm-scan, projects on disk, jobs and progress. To start with, it loads the PoC dump format (`out/poc/data/<clip>`) as the stand-in for a scanned project | core, media, ml |
| `mm-app` | The store: state tree, command dispatch, change notifications. Owns the engine, media and render pieces; draws into the host's texture; exports the C ABI (`include/mm_app.h`). Built as a `cdylib` in dev (the host loads it at start, so a Rust change needs no C++ relink) and as a `staticlib` for release | everything |
| `mm-cli` | Headless use of the same store: run commands, dump state, bench. For tests and CI without a window | mm-app |

A crate only depends on crates above it in this table. `mm-core` builds on any OS and is where most unit tests
live.

## The host (`native/app/`)

```
app/
  CMakeLists.txt
  src/main.cpp           start-up, flags, the QML engine, loads libmm_app
  src/store.{h,cpp}      Store: QObject with `state` (QVariantMap) and `call(cmd, args)`; applies Rust's change notes
  src/videoitem.{h,cpp}  QQuickRhiItem; hands Qt's MTLDevice, queue and texture to mm_app_render
  src/devtools.{h,cpp}   the developer hooks (below); only with --dev
  src/platform_mac.mm    macOS bits (title bar, appearance)
  qml/Main.qml           window, steps
  qml/pages/             Projects, Decisions, People, Editor, Export
  qml/components/        buttons, chips and fields, matching web/theme.css
  qml/Theme.qml          singleton: the tokens from web/theme.css
```

## Fast iteration

| change | what happens | target |
|---|---|---|
| QML | In dev, QML loads from `app/qml/` on disk. A file watcher reloads the window on save. The store lives in Rust, so state (open video, frame, selection) survives the reload | < 0.2 s |
| Rust | `cargo build -p mm-app` rebuilds the dylib, then `mmctl restart` relaunches the host with the same arguments and asks the store to restore its last state | a few seconds |
| C++ | ninja rebuild and relaunch. Rare, because the host is thin | seconds |

## Agent hooks (`devtools`, only with `--dev`)

An HTTP server on 127.0.0.1. Its port is written to `out/native/devtools.json`.
`native/tools/mmctl` wraps it for the command line.

| route | does |
|---|---|
| `GET /state[?path=a.b]` | the store's state as JSON |
| `POST /call {cmd, args}` | run a command, as the UI does |
| `GET /tree[?root=name&depth=n]` | QML items: type, objectName, text, scene rect, visible, enabled, value, checked, focus |
| `GET /find?text=…` or `?name=…` | items by visible text or objectName |
| `POST /click {name\|text\|x,y}`, `/drag`, `/key`, `/type` | real input events (mouse/key events sent to the window) |
| `POST /eval {js}` | evaluate JavaScript in the root QML context |
| `GET /grab[?name=item]` | PNG of the window or an item |
| `POST /reload` | reload QML now |
| `GET /logs?since=n` | qDebug, QML warnings and Rust logs, in order |
| `GET /wait?js=…&timeout=ms` | wait until an expression is true |
| `GET /perf` | frame times, composite times, decode times |
| `POST /quit`, `/restart` | quit, or relaunch with the same arguments |

**Privacy:** `/grab` greys out the video unless the open clip is a public-domain clip (`test_clips/jazz.mp4` and
the clips made from it in `out/poc/clips/`). An agent can then look at the UI around a real video without seeing
anyone in it (AGENTS.md).
