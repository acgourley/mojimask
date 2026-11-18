# How mojimask was ported to the native app, and why it is the way it is

**Goal:** the app in `native/` (Rust: wgpu, GStreamer, ort, Qt Quick) does everything the Python app does, and gives
the same answers on the same clips; then it replaces the Python app, on the Mac first, then Windows and Linux. Why this
stack: [stack-decision.md](stack-decision.md). How it is put together: [../ARCHITECTURE.md](../ARCHITECTURE.md). What
is not done yet: [../README.md](../README.md), "Known gaps". The port's history, step by step, is in the commit
messages.

## How we port: replace the old app piece by piece, with parity at every step

1. **Same project folder.** The Rust app reads and writes the same project files as the Python app:
   `facts.json`, `edits.json`, `proto/*`, `dets/*`, `mats_v2.npy` and the rest. So during the port both apps can
   open the same project, and every Rust module can be compared with its Python original on the same input.
2. **The UI first, over the Python engine.** The first usable Rust app is the new UI with the Python engine
   still behind it as a sidecar process. People get the native app early, and the engine is then replaced module
   by module underneath it, without UI changes.
3. **Prove parity on intermediate outputs, not just the end result.** Every ported engine module gets a parity
   test on jazz: run the Python and Rust versions on the same input, then diff their output files.
   - Exact where the maths allows: the ML PoC got bit-identical tensors by copying `cv2.resize`'s fixed-point
     maths.
   - A stated tolerance otherwise.
   - The rules flip on pixel noise (memory: "bench with --detect"), so check every stage, not only the end.
4. **The decisions benchmark is the final judge.** `tools/bench/decision_bench.py`'s numbers, on the Rust scan:
   `native/tools/parity/engine2_scan_bench.py`, and for rule changes `native/tools/bench/rule_bench.py`. A step is finished only when its numbers (asked, rule wrong,
   result) equal Python's on jazz, with depth on and off, and `photo_bench` agrees too.
5. **Agents work through the hooks.** Every UI step is checked with `mmctl` (tree, wait, grab on jazz), never
   left for a person to check by eye. The privacy rules in AGENTS.md apply unchanged.

## Decisions (user, 2026-11-17)

1. **Emoji:** Apple Color Emoji through CoreText for now (today's look). The atlas question comes back with Windows and Linux.
2. **OpenCV:** scope a port (below) before deciding.
3. **Release:** one release when the native app does everything; no interim release over the Python sidecar.
4. **The Python app is frozen** apart from fixes. Its behaviour is the parity target.
5. **Licence:** whatever is needed. The native app stays AGPL-3.0, which works with every dependency chosen so
   far: Qt LGPL, GStreamer LGPL, ONNX Runtime MIT, and the GPL/AGPL models.
6. **Motion:** our own, domain-specific motion (mm-motion) after an exact baseline; see the OpenCV scope below.
7. **Mac first** (2026-11-17): ship the Mac app before Windows and Linux.
8. **Fewer same-person questions** (2026-11-17): the native scan's question rules now differ from the frozen Python
   app on purpose (mm-scan questions.rs, mm-core facts.rs; the numbers are in their commit). Parity checks of the
   scan show those differences.

## Scope: porting the OpenCV calls to Rust

The engine calls OpenCV in about 50 places. **What matters is whether a call can change a decision.** The scan's
rules flip on pixel noise, so those calls must be bit-exact copies of OpenCV's algorithm, including its
fixed-point maths and its random number generator. The rest only draw pictures, and a GPU version within the drawing
tolerance is enough.

**Must be bit-exact (they feed the scan's decisions):**

| function | used by | what an exact port involves | size |
|---|---|---|---|
| `goodFeaturesToTrack` | track.py camera motion and follow; scan/follow.py | `cornerMinEigenVal` (Sobel 3x3, box filter, float), threshold, 3x3 dilate, sort by response (stable order), min-distance grid | ~250 lines |
| `calcOpticalFlowPyrLK` | track.py (camera motion, forward-backward check), follow | `buildOpticalFlowPyramid` (pyrDown 5x5 Gaussian, border handling), Scharr derivatives in int16, the LK iterations with OpenCV's 14-bit fixed-point bilinear weights, its stopping criteria, and the `minEigThreshold` status rule | ~500 lines |
| `estimateAffinePartial2D` (RANSAC) | track.py camera motion, scan/follow.py | `cv::RNG` (its multiply-with-carry generator and seeding), sample checks, iteration count update, inlier counting, then Levenberg-Marquardt refinement (`cv::LMSolver`) on the inliers | ~450 lines |
| `warpAffine` (8U, linear) | reid.py SFace alignment, follow.py templates, depth.py | OpenCV 5 uses float kernels, not the old `INTER_BITS` tables; the 8-wide blocks and the scalar tails compute coordinates differently | ~200 lines |
| `resize` (LINEAR, CUBIC, AREA, NEAREST) | detector inputs, reid crops, depth input, frame scaling | Fixed-point LINEAR and CUBIC, and AREA for integer factors, were already exact in the ML PoC (`poc/ml/src/pre.rs`, unpublished). Add AREA for non-integer factors (weights table) and NEAREST | ~250 lines more |
| `cvtColor` BGR→GRAY, BGR↔RGB | feature tracking, ML inputs | this cv2 build routes it through Carotene (NEON): 15-bit coefficients R 9798, G 19235, B 3735 | ~30 lines |
| `GaussianBlur` (8U and 32F) | depth.py, lighting estimate | `getGaussianKernel` (its fixed small-kernel tables, the sigma rule), separable fixed-point filter for 8U, borders | ~200 lines |
| `erode`, `copyMakeBorder`, `pointPolygonTest`, `invertAffineTransform`, `transform` | depth.py, behind.py, track.py | simple | ~150 lines |

**Can be approximate (they only draw; they moved to the GPU, mm-render):** the blur and pixelate styles,
`formats.py`'s blurred background, the head-pose `warpPerspective` (becoming a shader transform), the lighting
`filter2D` on sprites, and thumbnails and crops for the UI.

**Done (2026-11-17, `crates/mm-vision`):** every function in this table, with 0 differing pixels against the project's cv2 over 3,474 cases (filter2D with 50+ taps uses a DFT in cv2: within 3e-7). The project's cv2 (OpenCV 5.0, arm64) has Carotene on, which changes BGR2GRAY, area resize by 2x and 4x, and u8 linear resize. A fixed Rust implementation is also more deterministic across platforms than cv2 itself. 

**Total (as scoped):** about 2,000 lines of Rust, plus a test that runs each function against `cv2` (the project's venv) on
jazz frames and random inputs and requires identical output. The tricky part is LK plus RANSAC: OpenCV's exact
iteration and RNG behaviour. Everything else is mechanical. OpenCV is Apache-2.0, so a port with attribution is
fine.

**Revised after the user's input (2026-11-17): write our own motion, after an exact baseline.**

The user has found writing their own optical flow worthwhile elsewhere, because the right algorithm depends on
the domain. Here there are two domains:
- **camera motion:** one similarity transform per frame pair;
- **following one head through a gap.**

Neither needs general sparse flow. So:
1. **Baseline first, revised:** no OpenCV link. Each scan stage is checked with Python's own files as its input (`mats_v2.npy`, and recorded flow/follow calls replayed, as the model stream did for propagate), so the port is proved stage by stage without OpenCV. The original idea was to link the small OpenCV build (`opencv` crate) for `goodFeaturesToTrack`,
   `calcOpticalFlowPyrLK` and `estimateAffinePartial2D` only. The Rust engine then reproduces Python exactly,
   and the whole port is proved without inventing anything.
2. **Then replace them with our own motion ,** judged by quality, not by matching Python:
   - **Camera motion:** e.g. a coarse-to-fine dense or block search on the GPU (the frames are already there),
     with a robust similarity fit (iteratively reweighted least squares, deterministic, no RNG).
   - **Head following:** a template or flow search in the head's box, designed for heads (scale, turning, motion
     blur).
   - Corner finding and RANSAC probably disappear with this design. If a sparse step remains, write simple
     deterministic versions; matching OpenCV no longer matters.
3. **How the replacement is judged:**
   - **Known motion:** jazz frames warped by known transforms (the real truth, at any amount of motion, blur and
     noise), measured as recovered-transform error.
   - **`decision_bench` against the census:** no worse than the baseline on jazz.
   - **Numbers only on the user's new clips** as they arrive. Only jazz has a census, so don't tune on it alone.
   - Each change is a separate commit with before and after bench lines (AGENTS.md), not mixed into the port.

So no exact port of LK, RANSAC or corners. The other bit-exact items above (`warpAffine`, `resize`,
`GaussianBlur`, `cvtColor` and the small ones) are still ported, because they're mechanical and they keep
detector, re-ID and depth inputs identical.

**Superseded: a spike of about a day.**
1. Port `goodFeaturesToTrack`, `calcOpticalFlowPyrLK` and `estimateAffinePartial2D` exactly.
2. Rebuild `mats_v2.npy` (camera motion) for jazz in Rust, and require it identical to Python's.

If the spike passes, port the rest (mechanical) and ship without OpenCV. If LK or RANSAC can't be made exact in
the spike, link the small OpenCV that `packaging/build_opencv.sh` already builds, through the `opencv` crate,
for these few functions, and revisit later.

## Risks
- **The scan's rules are tuned on jazz and sensitive to small numeric differences.** Mitigation: exact numerics
  where possible, parity at each stage, and the bench as the final judge.
- **Mac zero-copy relies on GStreamer internals.** Mitigation: the automatic one-copy fallback, plus a test that
  fails loudly when the meta changes.
- **QML is a new UI language for the project.** The pages are rewritten, not moved. Mitigation: live reload, the
  agent hooks, and the pen.dev designs as the source.
- **Windows and Linux are unproven so far** (API signatures only). Mitigation: a tiny Windows build
  (decode plus draw) early, well before the full port.
- **Only jazz has a census.** Clips the user is gathering (memory: benchmark plan) should be added to the parity
  set as numbers-only tests as they arrive.
