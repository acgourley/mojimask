# Working on mojimask

How to run, change and measure mojimask from a checkout: for coding agents and the people working with them. What
the project is: [README](README.md). The rules for contributions: [CONTRIBUTING.md](CONTRIBUTING.md).

## Where things are

| | |
|---|---|
| `desktop.py`, `app_server.py`, `decisions_cli.py` | the ways to start it: the Mac app, the same app in a browser, answering Decisions from a program |
| `mojimask/` | the engine: detection (`pipeline.py`, `onnx_core.py`, `video_io.py`), the scan (`scan/`), answers (`facts.py`), the project server (`review_server.py`), drawing and export (`track.py`, `lighting.py`, `gpu_draw.py`, `formats.py`) |
| `web/` | the pages, their scripts and styles, fonts and pictures: everything the app's window shows |
| `tools/` | developer tools: the benchmarks (`bench/`, with their history in `bench/data/`), the census labeller (`label/`), jazz's census (`truth/jazz.json`) and the screen catalog (`catalog/`) |
| `experiments/` | finished research, with its own `requirements.txt`; nothing in the app imports it |
| `packaging/` | the Mac build and its disk image, icon and licences |
| `models/` | model weights (not in git; `models/README.md` says where each comes from) |
| `test_clips/` | `jazz.mp4`, the public-domain sample clip; put your own clips here (git-ignored) |
| `out/` | everything the app and tools write from a checkout (git-ignored) |

## Setup

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
./packaging/build_opencv.sh      # ~5 min; needs cmake + ninja (brew install cmake ninja)
```

- Use the project's Python: `.venv/bin/python …` (or `source .venv/bin/activate` per terminal). A bare `python`
  may not exist.
- OpenCV is built from source because pip's `opencv-python` bundles a GPL build of FFmpeg the app never uses.
- Model files go in `models/onnx/`, except the depth model, `models/depth-anything-v2-small/model.onnx`. Where each
  comes from and its licence: [models/README.md](models/README.md).

## Running the app

```bash
.venv/bin/python app_server.py --port 8800 --dev                                # http://localhost:8800
MOJIMASK_DEPTH=anything-small .venv/bin/python app_server.py --port 8801 --dev  # the same, with depth
.venv/bin/python desktop.py                                                     # in its own window
```

- If a server is already on 8800, it's probably the person you're working for: start yours on another port.
- **A running server keeps the code it started with.** Each open project runs its own `mojimask/review_server.py`, started
  when it's opened. After changing Python code, restart the app server. If a page "looks old", check how long the
  server has been up.
- Projects live in `out/projects/` from a checkout, `~/Library/Application Support/Mojimask/Projects` in the Mac app.
- Is depth on? Opening a project logs `partial occlusion on: anything-small` in `out/projects/<id>/review_server.log`.

| setting | what it does | default |
|---|---|---|
| `--dev` / `MOJIMASK_DEV=1` | developer tools: Clone fresh, Export labels, the full Decisions view, `/splash` | off |
| `MOJIMASK_DEPTH=anything-small` (or `off`) | the depth model: the scan's depth rule, covers partly hidden behind things, the editor's Depth view (D) | off from `app_server.py`; on in the Mac app |
| `MOJIMASK_PROJECTS=<dir>` | where projects live | `out/projects/` |
| `MOJIMASK_APP=1` | behave as the Mac app (its projects folder); `desktop.py` sets it | off |
| `MOJIMASK_GPU=0` | draw covers on the CPU instead of Metal | GPU |
| `MOJIMASK_CACHE=<dir>` | CoreML's compiled models | `~/Library/Caches/org.mojimask.app` |
| `MOJIMASK_PLAN=default\|gpu\|ane\|cpu` | which chip runs each detector (`onnx_core.PLANS`) | `default` |
| `MOJIMASK_FACE_POOL=<n>` | copies of the face models shared among the scan's threads | 3 |
| `MOJIMASK_PORT=<n>` | a fixed port for `desktop.py`'s server | any free port |
| `MOJIMASK_SELFTEST=1` (+ `_VIDEO`, `_DROP`, `_OUT`, `_REPORT`) | `desktop.py` runs a scan and an export by itself and writes a report | off |

## The app's look

- The pages are in `web/`. One stylesheet, `theme.css`, and one icon set, `icons.js`, are loaded by `app.html` and the three pages in its
  frames (`decisions.html`, `people.html`, `review.html`). Colours, buttons, chips and fields come from there:
  change them there, not page by page.
- The name and its wordmark are set only in `web/brand.js`; the pages and `mojimask/about.py` read it. Bundle ids, `MOJIMASK_*`
  and folder names are identifiers and don't follow it.
- Nothing is fetched while the app runs: fonts go in `web/fonts/`, pictures in `web/assets/`, icons are inline SVG.
- The Mac window (`desktop.py`) uses the page's bar as its title bar and opens on `mojimask/splash.py`'s picture. Check
  changes to those in the real window, not only a browser.
- **The screen catalog** (`tools/catalog/`): scripted paths through the app, run in WebKit (the Mac window's engine)
  on fresh copies of the jazz project. It's an integration test (`.venv/bin/python tools/catalog/run.py` ends 1 if a
  path doesn't get through) and a viewer (`out/catalog/<run>/index.html`): every step of each path at a 13-inch Air's
  window (1280 × 740) and the first-launch size (1280 × 880), as a screenshot or as live HTML where other words can be
  tried in place, with each piece of text on it, its `strings.js` key and how it fits (characters to spare, wraps,
  cut off, below the fold), and every key with the steps it shows on. Look at it when writing or translating the
  app's words; add a path (`tools/catalog/paths.py`) for each new flow. `--drafts [file]` runs it with other wording laid over
  today's (`out/strings_draft.json`; the format is in `tools/catalog/strings_record.js`). Setup, once:
  `uv pip install --python .venv/bin/python -r tools/catalog/requirements.txt` and
  `.venv/bin/python -m playwright install webkit`.

## Building the Mac app

```bash
./packaging/build_mac.sh        # ~1 min -> dist/Mojimask.app, dist/Mojimask.dmg
```

- Needs the setup above and the models in `desktop.CHECK_MODELS` plus the depth model; it stops at once if one is
  missing. Its last step runs the app with `--check` and passes only if the JSON report ends `"ok": true`.
- Expected, harmless warnings: `skipping data collection for module 'cv2'` and `could not resolve
  '@rpath/libswiftCoreMedia.dylib'`. The full log is `build/build.log`.
- With developer tools: `MOJIMASK_DEV=1 dist/Mojimask.app/Contents/MacOS/Mojimask`. Signed ad hoc only.

## Changing the scan's rules

- The scan is the `mojimask/scan/` package (`newalgo.py` runs it); the rules that settle questions are in `blips.py`,
  `overlaps.py`, `same.py`, `cover.py` and `mojimask/facts.py`. **Each module's docstring states its rules and the evidence
  for them:** read it before changing one, and keep it true after. Modules import each other as
  `from mojimask.scan import graph as G`; command lines run from the repo root as
  `.venv/bin/python -m mojimask.scan.graph ...`.
- **Measure before and after** with the decisions benchmark:

  ```bash
  .venv/bin/python tools/bench/decision_bench.py jazz --note "what changed"   # add --depth for depth on
  .venv/bin/python tools/bench/decision_bench.py --history
  ```

  It rescans a scratch copy (`out/bench/jazz/`; detection cached, `--detect` redoes it), plays the Decisions page
  through with the truth answering, and logs a line to `tools/bench/data/decisions_history.jsonl`. It
  reports **asked** (questions a person would answer), **rule wrong** (rules that settled a question against the
  truth: `face uncovered` is the worst, then `needless cover`, `double cover`, `missed join` / `wrong join`) and
  the **result** (faces never boxed or left uncovered, people split or mixed). Look at the pictures behind any number
  that moved (`--sheets`, in `out/bench/jazz/sheets/`) before trusting it.
- **Photos** are clips of one frame (`video_io`). When changing the blip rules, also run
  `tools/bench/photo_bench.py jazz` (jazz's census frames as photos).
- **Truth:** `tools/truth/jazz.json`, the census (every face boxed by hand on every 4th frame). Only jazz has one, so
  don't tune rules on any other clip. Not `out/probe/gt_*.json`: by eye many of those "faces" are hands. A census
  of another clip is made with
  `.venv/bin/python tools/label/serve.py out/projects/<project> --name <clip> --every 4`.
- **Apply a rule change to an existing project** (answers kept):
  `MOJIMASK_DEPTH=anything-small .venv/bin/python tools/bench/refresh.py out/projects/<id>`, then restart the
  app server. It refuses if the boxes themselves would change (then the project needs a rescan, and loses its
  answers). **Clone fresh (test)** copies a project without your answers, as a new user would see it; it doesn't
  rerun the rules, so refresh the source first.
- Raise `newalgo.SCAN_VERSION` only when a change can renumber or reshape boxes (it throws away every project's
  answers).
- Keep the module's docstring true: the rule, and the evidence for it (the clip, the numbers, the date). Put the
  benchmark's before and after lines in the commit message.

## The Decisions page

- In the app it's the **Decisions** step (in a frame): "To answer" / "Answered". That is what people see.
  `/p/<id>/decisions.html` opened directly is the developer view: every record, filters, rule answers.
- **Answer from the command line** instead of clicking (the app must be running; JSON out; `--help` explains each
  question kind):

  ```bash
  .venv/bin/python decisions_cli.py projects
  .venv/bin/python decisions_cli.py questions <project>          # --all includes answered and rule-settled ones
  .venv/bin/python decisions_cli.py answer <project> <id> "not a face"
  .venv/bin/python decisions_cli.py clear <project> <id>         # puts the facts back as they were
  ```

  It goes through the running app exactly as a click does. Answers are tagged `via: "agent"`. One answer can settle
  or open others, so list the questions again after each. Work on a clone, never on someone's own project.

## Native app (native/)

The Rust rewrite (wgpu, GStreamer, ONNX Runtime, Qt Quick): [native/README.md](native/README.md) for building,
running and the iteration loop; [native/ARCHITECTURE.md](native/ARCHITECTURE.md) for how it fits together. All state
and logic are in Rust; QML only binds to `store.state` and calls `store.call(...)`.

- Build with `native/build.sh`; drive the running app with `native/tools/mmctl` (`mmctl help`). It prints JSON.
- **Checking a UI change:** start the app on jazz
  (`native/tools/mmctl start --video out/poc/clips/jazz_4k.mp4 --data out/poc/data/jazz_4k`), edit the QML in
  `native/app/qml` (the window reloads itself; the open video, frame and selection stay), then
  `mmctl wait '<JS that is true once your change shows>'`, `mmctl tree` / `mmctl find` to check items and their
  rects, and `mmctl grab out/native/<name>.png` to look at it. Give every interactive item an `objectName`.
- A Rust change: `mmctl rebuild` (about 2 s, state restored). A C++ change: `cmake --build native/app/build`, then
  `mmctl restart`.
- `mmctl grab` greys every picture unless the open clip is public domain (`test_clips/jazz.mp4`, `out/poc/clips/`),
  and `readback` refuses other clips: look at the UI around other videos, never at the people in them. If the person
  you work for has the app open on their own project, don't drive their window: start a second one with its own
  `--root` (a folder of symlinks to `web`, `models`, `test_clips`, `native`, with its own `out/`) and a copy of mmctl
  beside it.
- The app's fonts, words (`strings.js`), icons, brand, palettes and pictures are in `native/web/`; the Python app's
  `web/` links to them, so a change there shows in both apps.
- Scaled pictures in QML set `sourceSize` to the size they're drawn at (`components/Picture.qml` says why).
- **The scan's rules differ from the Python app's on purpose** (fewer same-person questions; `native/docs/porting.md`,
  decision 8). Measure a rule change with `native/tools/bench/rule_bench.py --note "..."` (jazz against its census,
  the party clips' worst case, depth on), before and after, and put both in the commit. A whole scan against Python's,
  numbers only, on any clip: `native/tools/parity/clip_scan_parity.py <video>`; the per-module parity checks are in
  `native/tools/parity/`.
- **The Mac app:** `native/packaging/make_app.sh --dmg` -> `native/dist/Mojimask.app` and `Mojimask.dmg` (the disk
  image as `packaging/make_dmg.sh` makes the Python app's). It fails unless the bundle's `--check` (a scan and an
  export of jazz with Homebrew hidden) passes. Signed ad hoc; `MOJIMASK_SIGN_ID` and `MOJIMASK_NOTARY_PROFILE` sign
  with a Developer ID and notarize.

## Privacy (hard rules)

- People's faces never leave the machine: no cloud vision, no sending frames or crops anywhere (including to
  yourself through an API), never commit clips, frames or face images. `out/`, videos and pictures are git-ignored,
  and `.githooks/check_media.py` refuses them; don't force-add them or skip the hook.
- `test_clips/jazz.mp4` is a public-domain silent film: its frames and crops may be looked at directly, and should
  be, before trusting a measurement. Every other clip or photo shows real people: numbers only, until it is listed
  here as public domain. In docs, comments and commit messages, never describe how anyone in them looks: say
  "guest 3", not their clothes.
- Downloads (models, packages) need a go-ahead from the person you're working for.

## Notes, commits and pull requests

- Why a change was made goes in its commit message (and its pull request): what changed, what was measured
  before and after, what you looked at, what you didn't do and why. Why a rule is what it is goes in its module's
  docstring. There is no separate working log. The README is for people using the app: no working notes there.
- Commit only when asked. End each commit message you wrote with an `Assisted-by:` trailer naming the tool and
  model, e.g. `Assisted-by: Claude Code (claude-opus-5-5)`, not `Co-Authored-By:`: a person is the author.
- A human submits each change and answers for it ([CONTRIBUTING.md](CONTRIBUTING.md)): keep changes small enough
  to review, and don't open issues, pull requests or comments on your own.
