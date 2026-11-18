# Mojimask native

The Rust rewrite of mojimask: Rust (wgpu, GStreamer, ONNX Runtime) behind a thin Qt Quick host. Why this stack:
[docs/stack-decision.md](docs/stack-decision.md). How it fits together: [ARCHITECTURE.md](ARCHITECTURE.md). How it was
ported, the user's decisions and the OpenCV scope: [docs/porting.md](docs/porting.md). For coding agents and the people
working with them.

The Python app (`mojimask/`, `web/`) is still the product. This one runs the whole journey on its own Rust engine
(`--engine native`, the default; the Python engine stays behind `--engine python` for comparison): **Projects**
(import, scan, "Deciding who's who"), the **result** after a first scan, **Decisions**, **People**, the **Editor** (the
export's own covers drawn locally by mm-render, every editing tool of web/review.html) and **Export** (formats, trim,
the corner mark, higher-risk mode, encoded through mm-media).

## Build

Needs Rust 1.95 (`rust-toolchain.toml`; rustup fetches it), Homebrew's Qt 6.12 (with QtHttpServer and QtTest),
GStreamer 1.28, cmake and ninja. From the repo root:

```bash
native/build.sh            # dev: the store dylib + mm-cli (cargo), then the Qt host (cmake + ninja)
native/build.sh --release  # the store linked statically: native/app/build-release/Mojimask.app
```

Build output goes to `native/target/` (cargo) and `native/app/build*/` (cmake); both are git-ignored.
The first cargo build takes a few minutes; after that a change to `mm-app` rebuilds in about a second.

## Run

```bash
native/tools/mmctl start --video out/poc/clips/jazz_4k.mp4 --data out/poc/data/jazz_4k   # dev, in the background
native/app/build/Mojimask.app/Contents/MacOS/Mojimask --dev --video test_clips/jazz.mp4 --data out/poc/data/jazz
```

| flag | does |
|---|---|
| `--dev` | QML from `native/app/qml` on disk, reloaded on save; the developer hooks on 127.0.0.1:9200-9299 |
| `--video CLIP`, `--data DUMP` | open a video, with cover data: a PoC dump folder (`out/poc/data/<clip>`, made by `native/tools/dump_reference.py`), the stand-in for a scanned project until the scan is ported |
| `--restore` | reopen what the store saved in `out/native/state.json` (`mmctl restart` passes it): the open project (once the engine is up), its page and frame; or a `--video` and its edits |
| `--lib PATH` | the store dylib to load (default `native/target/debug/libmm_app.dylib`) |
| `--engine native\|python` | the engine behind the pages: the Rust one in this process (default) or app_server.py as a sidecar |
| `--motion python\|native` | the native engine's scan motion (and a placed cover's flow): Python's cv2 through the project's venv (default for now: native motion asks one extra question on jazz; switching is the user's call) or mm-motion (no Python at all). Shown in the dev bar |
| `--projects DIR` | the projects folder (default: `out/native/projects` in a checkout, `~/Library/Application Support/Mojimask/Projects` bundled; `MOJIMASK_PROJECTS`) |
| `--python PATH` | the Python for `--engine python` and `--motion python` (default: `MOJIMASK_PYTHON`, the checkout's venv) |
| `--no-sidecar` | don't start an engine (the Editor alone, on a PoC dump) |
| `--check [DIR]` | the self-test: the native engine scans `test_clips/jazz.mp4` (motion native, depth on when the model is there), the store exports it with covers; one JSON line ending `"ok": true` |
| `MOJIMASK_DEPTH=anything-small` | passed to the engine: the depth model on (the covers' depth cut, the Editor's Depth view, D) |

**Where files are** (`mm_core::paths`): the read-only root (`web/`, `models/`, `test_clips/`: the checkout, or the
bundle's `Contents/Resources`; `MM_ROOT`, `MOJIMASK_MODELS`) and the data folder (`state.json`, projects: `out/native/`
in a checkout, `~/Library/Application Support/Mojimask` bundled; CoreML's compiles in
`~/Library/Caches/org.mojimask.app/coreml`). Inside a bundle the host sets GStreamer's and GIO's environment itself
(main.cpp `bundleEnv`). No build-machine path is compiled into a release binary (the main checkout's models and venv
are a debug-build fallback only).

## The native engine (`--engine native`)

`mm_engine::native` answers the same routes as app_server.py and review_server.py, in this process
(`crates/mm-engine/src/native/`; the app talks to either through `mm_engine::backend::Backend`):
- the library, import, the scan (`mm_engine::scan` on a thread: its reports and "Deciding who's who" events live),
  cancel, clear cache, delete, clone fresh, relink, the pictures (thumb, preview, head);
- per open project, one `mm_core::edits::Project`: the single writer of edits.json and facts.json, holding
  `.server.lock` as review_server does (a Python server can't open the project meanwhile, and the reverse); ops,
  facts, decisions (with the "whose face is this?" questions, `native/whose.rs`), undo / redo, split, propagate and
  undo correction (`native/follow.rs`: review_server.flow_step through Python with `--motion python`, mm-motion with
  `native`); the resolved covers (`state`), boxes, poses, formats, export labels;
- draw records from mm-records, handed to the Editor and the export directly (no JSON).
Pictures QML loads by URL (frames, thumbnails) come from a small HTTP server of its own on 127.0.0.1:9300-9399.
Not yet: choosing a portrait photo (`POST /api/portrait`).

Checked: `native/tools/parity/appnative_journey.py` drives the whole journey through mmctl with either engine, and
`--compare` checks edits.json byte for byte, facts.json except its time stamps, and the exports frame by frame.

## The Python engine (sidecar, `--engine python`)

The app starts `app_server.py` unchanged (`crates/mm-engine/src/python/sidecar.py`) on a port in 9300-9399 and talks to
it over HTTP, as `web/app.html` does. Its log is **`out/native/sidecar.log`**; each open project's server logs to
`out/native/projects/<id>/review_server.log`. Each project server is `native/tools/sidecar/project_server.py`:
`mojimask/review_server.py`, unchanged, plus three routes for the native app:
- `/api/native/setup` and `/api/native/records?from=&to=&style=&wait=&strict=`: every frame's draw records (the head box,
  the emoji or portrait, head pose, the lighting estimate and velocity, the depth cut), decided exactly as the export
  decides them (`review_server.render_setup`, `track.draw_order`, `Occlusion.plan`). The app draws them with mm-render's
  `Drawer`, in the Editor and in the export (`crates/mm-app/src/records.rs`); lighting's grain is measured on the frame.
- `/api/native/export_frame?t=&preset=&fit=&mark=&strict=`: frame t as Python's export draws it (PNG), the parity
  reference.

**One writer.** A project's `edits.json` and `facts.json` belong to its review server (it holds `.server.lock`). The
app never writes them: every edit goes to that server as ops, facts or its own rules (`/api/ops`, `/api/facts`,
`/api/propagate`, `/api/split_cover`, `/api/undo_correction`), and its reply is taken as the truth
(`crates/mm-app/src/editor.rs`). Undo and redo are the server's history, shared by every page.

The covers only draw when the dump was made for a video of the same size (`jazz_4k` dump on `jazz_4k.mp4`, `jazz`
dump on `test_clips/jazz.mp4`); otherwise you see the video alone and the status line says so.

## Layout

| | |
|---|---|
| `crates/mm-core` | plain data and rules: time (frame t <-> source pts), geometry, cover rows, edits, colour, the privacy allowlist. Unit tests live here |
| `crates/mm-media` | GStreamer: `Decoder` (probe, seek, next on the slot timeline), frame import into wgpu (`mac/cv.rs`: zero-copy), `Exporter` (vtenc + mp4mux, audio copied) |
| `crates/mm-render` | wgpu: own or adopted (Qt's) device, the WGSL compositor, YUV->RGB, prep, NV12, readback |
| `crates/mm-ml` | `ort`: sessions, execution providers, pre/post-processing; `mm-ml-parity` |
| `crates/mm-engine` | projects (for now the PoC dump + edits), jobs and progress; the scan goes here |
| `crates/mm-app` | the store (state tree, commands, change notes), the preview decoder and player, rendering into the host's texture, export; the C ABI `include/mm_app.h` |
| `crates/mm-cli` | the store without a window |
| `app/src` | the C++ host: `main.cpp` (start-up, live reload), `store` (bridge), `videoitem` (QQuickRhiItem), `devtools` (hooks), `platform_mac.mm` |
| `app/qml` | `Main.qml`, `Theme.qml` (web/theme.css's tokens), `components/`, `pages/` |
| `web/` | what the app shows and draws with: fonts, the words (`strings.json`), icons, the brand, emoji palettes and eyes, the paintings and the default portrait (`mm_core::paths::web`; the bundle's `Resources/web`). The Python app's `web/` links to these until it is retired |
| `packaging/` | the Mac bundle and disk image (`make_app.sh`, `make_dmg.sh`), the icon and the disk image's picture drawn at build time (`make_icon.swift`, `dmg_background.swift`), licence texts (`licenses/`, `notices.py`) |
| `tools/mmctl` | the hooks from the command line |

## How it works, in one paragraph

All state is in Rust (`mm-app`), one JSON-serialisable tree. QML reads it (`store.state.editor.t`) and changes it only
with `store.call("seek", {t: 40})`; the hooks and `mm-cli` call the same commands. When a command or a worker thread
changes something, Rust names the paths that changed and the host re-reads the state; `"@render"` makes the video
item draw. On Qt's render thread the item hands Qt's MTLDevice, command queue and its texture to `mm_app_render`:
Rust adopts the device once and composites with wgpu on Qt's own queue (no CPU copies: the decoder's IOSurface planes
are imported directly). New commands and state fields go through the C ABI as JSON, so adding a feature never touches
C++. Commands are listed by `mmctl call help`.

## The iteration loop

| change | do | measured (M2 Ultra, 2026-11-17) |
|---|---|---|
| QML (`app/qml/**`) | save; the window reloads by itself, the store's state (open video, frame, selection, edits) stays | 0.11-0.16 s save to first frame |
| Rust | `native/tools/mmctl rebuild` (cargo builds the dylib, the app restarts with `--restore`) | ~2 s (cargo 1 s, restart 1 s) |
| C++ (`app/src`) | `cmake --build native/app/build`, then `mmctl restart` | ~2 s build + 1 s restart |
| cold start | `mmctl start ...` | 0.7 s to the hooks, 0.9 s to the first 4K frame drawn |

## Checking a change: mmctl

`native/tools/mmctl` (Rust, `crates/mm-ctl`, built by the script when needed) talks to the running app through its
hooks. Every command prints JSON; it exits 1 when the app says no. `mmctl help` lists everything.

```bash
M=native/tools/mmctl
$M start --video out/poc/clips/jazz_4k.mp4 --data out/poc/data/jazz_4k
$M tree --root page-editor --depth 4     # items: type, objectName, text, scene rect, visible, enabled, value
$M find Play                             # by visible text (or --name play-button)
$M click Play                            # a real mouse click at its centre
$M wait 'store.state.editor.playing'     # until a JS expression (root QML context) is true
$M key Space                             # real key events: Space, Right, Ctrl+E
$M call seek '{"t": 40}'                 # any store command, as the UI calls it
$M wait 'store.state.editor.shown === 40'
$M grab out/native/shot.png              # the window as a PNG (--name ITEM for one item)
$M drag --name cover-3 --dx 60 --dy 20   # press, move, release
$M state editor.covers                   # the store's state, or part of it
$M call export '{"out": "out/native/jazz_4k.mp4"}'   # a project: also preset, fit, start, end, mark, strict
$M wait 'store.state.export.state !== "running"' --timeout 60000
$M logs --since 0                        # Qt, QML and Rust log lines, in order
$M perf                                  # frame, composite, decode and QML reload times
$M rebuild                               # after a Rust change
$M quit
```

- Every interactive item in the QML has an `objectName`; keep it that way, so the hooks can find what you add.
- The hooks: `GET /state`, `POST /call`, `GET /tree`, `/find`, `POST /click`, `/drag`, `/key`, `/type`, `/eval`,
  `GET /grab`, `POST /reload`, `GET /logs`, `/wait`, `/perf`, `POST /quit`, `/restart`. The port is in
  `out/native/devtools.json`.
- **Seeing the drawn frame exactly:** `mmctl call readback '{"path": "out/native/f.png"}'` saves the video item's texture
  at full resolution the next time it's drawn (`store.state.readback.state` turns `"done"`). Compare with the
  reference: `<project venv>/bin/python native/tools/compare_frames.py out/poc/data/jazz_4k/ref <dir> --json`.
  Without a window: `native/target/debug/mm-cli --video out/poc/clips/jazz_4k.mp4 --data out/poc/data/jazz_4k call seek '{"t":40}' render out/native/f.png`.

### A project, by commands

```bash
$M call open_project '{"id": "jazz"}'          # its result after a first scan, else Decisions or the Editor
$M wait 'store.state.editor.records > 0'      # the Editor has its draw records
$M call editor_click '{"x": 182, "y": 107}'   # select the box there (video px); editor.view.sel says what's selected
$M call editor_key '{"key": "E"}'             # the Editor's keys: E U S [ ] K R V D B Del Esc Left Right Space
$M call editor_draw '{"x0": 300, "y0": 350, "x1": 350, "y1": 400}'   # drag out a box: a cover added by hand
$M call editor_style '{"style": "pixelate"}'  # also editor_light, editor_palette, editor_head_pose, editor_toggle
$M call undo
$M call go '{"page": "decisions", "piece": "10"}'   # only box 10's decisions (the Editor's R, People's Why)
$M call edit_at '{"t": 63, "piece": "10"}'    # a picture clicked in Decisions or People
$M call export_set '{"preset": "vertical", "fit": "fit", "mark": true}'
$M call export_run '{"out": "out/native/v.mp4"}'
$M call set_dev '{"on": false}'               # the pages as someone without --dev sees them (the hooks stay)
```

Real key presses (`mmctl key E`) reach the pages' shortcuts only while the window is the active one; `editor_key`
runs the same code without it.

**Parity** (the app running, a public-domain project open in the Editor):
```bash
<venv>/bin/python native/tools/sidecar/parity.py editor --frames 0,40,80,126       # readback vs Python's export frame
<venv>/bin/python native/tools/sidecar/parity.py export OUT.mp4 --py-export PY.mp4  # H.264 vs H.264 (POST /api/render)
```

## Privacy (AGENTS.md)

Faces never leave the machine. Here that means:
- `/grab` paints every video item grey unless the open clip is public domain: `test_clips/jazz.mp4` or a clip under
  `out/poc/clips/` (`mm_core::privacy`, by real path, so a copy elsewhere doesn't count). The rest of the window
  stays visible, so you can check the UI around anyone's video without seeing them.
- `readback` and `mm-cli render` refuse other clips.
- Work on jazz. Never open `out/projects/` or other clips in `test_clips/` to look at them; pictures you write go
  under `out/`.

## Tests

```bash
cd native && cargo test                                                   # mm-core's rules, the store's helpers
native/target/release/mm-ml-parity out/poc/results/ml/ref out/native/ml_parity.json   # ML outputs vs Python (CPU: exact)
```

## Known gaps

- **Not ported yet:** choosing a new portrait photo (`POST /api/portrait`); the export preview's blurred "fit"
  background; the People page's "ask" filter; the depth view's face-zone circle.
- **The scan's motion:** `--motion native` (no Python) asks one extra question on jazz; from a checkout `python` is the
  scan's default, the bundled app uses `native`.
- **Seeking** an iPhone clip takes about 1.5 s (long GOP), so scrubbing is slow; forward decoding runs at 440 fps.
- **The lighting estimate** (about 80 s on a 50 s 4K clip) shows no progress, and all records are rebuilt every 2 s
  while it runs.
- **The video in the Editor** is a 4K texture shrunk into the item without mipmaps (aliasing on detail); it should be
  drawn at the item's size.
- **The corner mark** is a few levels off Python's on its letters (max error ~70 on a few pixels), so a marked export is
  ~0.5 dB further from Python's than Python's own H.264 is.
- **Export** decodes and converts each frame on the CPU and reads the drawn frame back: fine for 1080p, slow-ish for 4K.
- **A portrait's own pose** differs slightly from Python's (the `image` crate's Lanczos vs PIL): roll 0.16°, same facing.
- **Packaging:** launcher.c, check.sh and mm-cli could leave the bundle now that `mojimask-app --check` exists (about
  -30 MB). Signed ad hoc, like the Python app; Developer ID signing and notarizing are ready in make_app.sh but untried.
- **Mac only:** the host renders through Metal; Windows (D3D12) and Linux (Vulkan) need their own adopt and import code.
- meta.json's `engine` block has a different key order and version string from Python's (cosmetic).

## Bugs found in the Python app while porting

The Python app is frozen apart from fixes; these are fix candidates (fix Python first, then the port, which keeps
Python's behaviour for parity).

1. **`pose_util.facing` never returns "front":** it adds two numpy bools (a logical or), so `eyes == 2` is never true.
   Only the review server's skeleton view is affected.
2. **RetinaFace on CoreML MLProgram (`MOJIMASK_PLAN=gpu`) is off by up to 46% of the output range**, even with pinned
   sizes. The default plan runs it on the ANE, which is fine; `gpu` should probably not be offered until this is
   understood.
3. **A possible 35 s hang at start-up** from HTTPServer's reverse-DNS lookup of 127.0.0.1 (seen in a PoC's own server):
   check `app_server` and `review_server`.
4. **The attach rule can move a box to a person who isn't on screen** (jazz, with our own camera motion: two
   single-frame boxes on a man's head at frames 45 and 50 went to a person absent from 22 to 80, projected across the
   gap). It flips on motion noise. Proposed rule (Python first, decision_bench before and after): when a fragment's box
   overlaps a piece on screen at that frame, prefer that piece over one projected across a gap. The full native-motion
   scan doesn't split anyone; its bench equals Python's but for one extra question.
