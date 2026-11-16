# mojimask

Cover every face in a party video or photo with an emoji (or blur, pixelate, a solid box, or one photo), for the
footage where ordinary face blurring fails: a handheld phone, a moving crowd, strobing coloured light,
motion blur, costumes. It finds and follows the faces, asks you about the moments it isn't sure of,
and exports a clean video shaped for social media.

Everything happens on your computer. No video, frame or face is uploaded anywhere.

Free and open source (AGPL-3.0). Status: v0.1, Mac only (Apple silicon, macOS 13 or later).

## Using it

Unzip `mojimask-mac.zip` and follow `Read me first.txt` inside it. In short: choose a video, wait for the scan,
check the result, export. The app is signed ad hoc, so macOS asks you to confirm it once in System Settings; the
read-me shows how.

## Building it

Needs an Apple-silicon Mac with Python 3.11, [uv](https://docs.astral.sh/uv/), cmake and ninja
(`brew install cmake ninja`).

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
./packaging/build_opencv.sh      # a small OpenCV, ~5 min
# put the model files in models/: see models/README.md
.venv/bin/python desktop.py      # run it from source
./packaging/build_mac.sh         # or build it: dist/Mojimask.app and dist/mojimask-mac.zip
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), including how to work with AI coding tools here. How to run, change and
measure the app: [AGENTS.md](AGENTS.md), written for coding agents and people alike.

## Licence

AGPL-3.0 (see `LICENSE`), which the bundled YOLO11 pose model (Ultralytics, AGPL-3.0) requires. Model
weights have their own licences, listed in `models/README.md`; everything else bundled is in
`THIRD_PARTY_NOTICES.md`.
