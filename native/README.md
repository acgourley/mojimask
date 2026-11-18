# Mojimask native

mojimask, the app: Rust (wgpu, GStreamer, ONNX Runtime) behind a thin Qt Quick host. Why this stack:
[docs/stack-decision.md](docs/stack-decision.md). How it fits together: [ARCHITECTURE.md](ARCHITECTURE.md). How it was
ported from the first (Python) version, the user's decisions and the OpenCV scope: [docs/porting.md](docs/porting.md).
For coding agents and the people working with them.

The whole journey runs on its own Rust engine, in the app's process: **Projects** (import, scan, "Deciding who's
who"), the **result** after a first scan, **Decisions**, **People**, the **Editor** (the covers drawn by mm-render)
and **Export** (formats, trim, the corner mark, higher-risk mode, encoded through mm-media).

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
native/tools/mmctl start                                                  # dev, in the background
native/app/build/Mojimask.app/Contents/MacOS/Mojimask --dev               # dev, in the foreground
```

| flag | does |
|---|---|
| `--dev` | QML from `native/app/qml` on disk, reloaded on save; the developer hooks on 127.0.0.1:9200-9299 |
| `--video CLIP`, `--data DUMP` | open a video alone in the Editor, with cover data from a dump of the Python app's covers (`out/poc/data/<clip>`, from before the port), for working on drawing without a scan |
| `--restore` | reopen what the store saved in `out/native/state.json` (`mmctl restart` passes it): the open project (once the engine is up), its page and frame; or a `--video` and its edits |
| `--lib PATH` | the store dylib to load (default `native/target/debug/libmm_app.dylib`) |
| `--projects DIR` | the projects folder (default: `out/native/projects` in a checkout, `~/Library/Application Support/Mojimask/Projects` bundled; `MOJIMASK_PROJECTS`) |
| `--no-sidecar` | don't start the engine (the Editor alone, on a dump) |
| `--check [DIR]` | the self-test: the engine scans `test_clips/jazz.mp4` (depth on when the model is there), the store exports it with covers; one JSON line ending `"ok": true` |
| `MOJIMASK_DEPTH=off` | the depth model off (it is on whenever installed: the scan's depth rule, the covers' depth cut, the Editor's Depth view, D) |

**Where files are** (`mm_core::paths`): the read-only root (`native/web/`, `models/`, `test_clips/`: the checkout, or the
bundle's `Contents/Resources`; `MM_ROOT`, `MOJIMASK_MODELS`) and the data folder (`state.json`, projects: `out/native/`
in a checkout, `~/Library/Application Support/Mojimask` bundled; CoreML's compiles in
`~/Library/Caches/org.mojimask.app/coreml`). Inside a bundle the host sets GStreamer's and GIO's environment itself
(main.cpp `bundleEnv`). No build-machine path is compiled into a release binary (in a git worktree, debug builds use
the main checkout's models).

## The engine

`mm_engine::native` answers the pages' routes in this process (`crates/mm-engine/src/native/`; the app talks to it
through `mm_engine::backend::Backend`, the routes the Python app's pages used):
- the library, import, the scan (`mm_engine::scan` on a thread: its reports and "Deciding who's who" events live),
  cancel, clear cache, delete, clone fresh, relink, the pictures (thumb, preview, head);
- per open project, one `mm_core::edits::Project`: the single writer of edits.json and facts.json, holding
  `.server.lock`; ops, facts, decisions (with the "whose face is this?" questions, `native/whose.rs`), undo / redo,
  split, propagate and undo correction (`native/follow.rs`, mm-motion), the portrait photo, the resolved covers
  (`state`), boxes, poses, formats, export labels;
- draw records from mm-records, handed to the Editor and the export directly (no JSON).
Pictures QML loads by URL (frames, thumbnails) come from a small HTTP server of its own on 127.0.0.1:9300-9399.

## Layout

| | |
|---|---|
| `crates/mm-core` | plain data and rules: time (frame t <-> source pts), geometry, cover rows, edits, colour, the privacy allowlist. Unit tests live here |
| `crates/mm-media` | GStreamer: `Decoder` (probe, seek, next on the slot timeline), frame import into wgpu (`mac/cv.rs`: zero-copy), `Exporter` (vtenc + mp4mux, audio copied) |
| `crates/mm-render` | wgpu: own or adopted (Qt's) device, the WGSL compositor, YUV->RGB, prep, NV12, readback |
| `crates/mm-ml` | `ort`: sessions, execution providers, pre/post-processing |
| `crates/mm-engine` | the scan (`scan.rs`, detection, depth), the project server (`native/`), jobs and progress |
| `crates/mm-app` | the store (state tree, commands, change notes), the preview decoder and player, rendering into the host's texture, export; the C ABI `include/mm_app.h` |
| `crates/mm-cli` | the store without a window; `mm-cli scan`, `mm-cli decisions` |
| `crates/mm-vision`, `mm-analysis`, `mm-motion`, `mm-scan`, `mm-records` | OpenCV's calls the scan needs (bit-exact), the face, body, pose and depth models' evidence, our own camera motion and head following, the scan's rules, the covers to draw |
| `crates/mm-ctl`, `mm-pack`, `mm-bench` | mmctl, the packaging steps, the rules' benchmark |
| `app/src` | the C++ host: `main.cpp` (start-up, live reload), `store` (bridge), `videoitem` (QQuickRhiItem), `devtools` (hooks), `platform_mac.mm` |
| `app/qml` | `Main.qml`, `Theme.qml` (the colours, type and sizes), `components/`, `pages/` |
| `web/` | what the app shows and draws with: fonts, the words (`strings.json`), icons, the brand, emoji palettes and eyes, the paintings and the default portrait (`mm_core::paths::web`; the bundle's `Resources/web`) |
| `packaging/` | the Mac bundle and disk image (`make_app.sh`, `make_dmg.sh`), the icon and the disk image's picture drawn at build time (`make_icon.swift`, `dmg_background.swift`), licence texts (`licenses/`, gathered by `mm-pack notices`) |
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
$M start --video out/poc/clips/jazz_4k.mp4
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
  at full resolution the next time it's drawn (`store.state.readback.state` turns `"done"`). Without a window:
  `native/target/debug/mm-cli --video out/poc/clips/jazz_4k.mp4 call seek '{"t":40}' render out/native/f.png`.

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

## Privacy (AGENTS.md)

Faces never leave the machine. Here that means:
- `/grab` paints every video item grey unless the open clip is public domain: `test_clips/jazz.mp4` or a clip under
  `out/poc/clips/` (`mm_core::privacy`, by real path, so a copy elsewhere doesn't count). The rest of the window
  stays visible, so you can check the UI around anyone's video without seeing them.
- `readback` and `mm-cli render` refuse other clips.
- Work on jazz. Never open projects of real footage or other clips in `test_clips/` to look at them; pictures you
  write go under `out/`.

## Tests

```bash
cd native && cargo test                                  # the rules, the store's helpers, the portrait route
cargo run -q --release -p mm-bench -- --note "..."       # the scan's rules against jazz's census (AGENTS.md)
native/packaging/make_app.sh                             # the Mac app; its --check scans and exports jazz
```

## Known gaps

- **Not ported yet:** the export preview's blurred "fit" background (the preview shows a dim copy; the exported video
  has the blur); the People page's "ask" filter; the depth view's face-zone circle.
- **Seeking** an iPhone clip takes about 1.5 s (long GOP), so scrubbing is slow; forward decoding runs at 440 fps.
- **The lighting estimate** (about 80 s on a 50 s 4K clip) shows no progress, and all records are rebuilt every 2 s
  while it runs.
- **Export** decodes and converts each frame on the CPU and reads the drawn frame back: fine for 1080p, slow-ish for 4K.
- **Packaging:** launcher.c, check.sh and mm-cli could leave the bundle now that `mojimask-app --check` exists (about
  -30 MB). Signed ad hoc, like 0.1; Developer ID signing and notarizing are ready in make_app.sh but untried.
- **Mac only:** the host renders through Metal; Windows (D3D12) and Linux (Vulkan) need their own adopt and import code.

## Known issues in the rules, carried over from the first version

The port kept the Python app's behaviour exactly, including these; each is a candidate for its own change, measured
with the rules' benchmark before and after.

1. **The head's facing is never "front"** (`mm-analysis` pose_util `facing`, as Python's `pose_util.facing`): two
   booleans are added, which is a logical or, so `eyes == 2` is never true and a face with both eyes and the nose
   visible counts as "profile". Only the developer skeleton view uses it.
2. **RetinaFace on CoreML's MLProgram (`MOJIMASK_PLAN=gpu`) is off by up to 46% of the output range**, even with pinned
   sizes. The default plan runs it on the ANE, which is fine; `gpu` should not be offered until this is understood.
3. **The attach rule can move a box to a person who isn't on screen** (jazz: two single-frame boxes on a man's head at
   frames 45 and 50 went to a person absent from frame 22 to 80, projected across the gap). It flips on motion noise.
   Proposed: when a fragment's box overlaps a piece on screen at that frame, prefer that piece over one projected
   across a gap.
