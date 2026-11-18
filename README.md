<p align="center">
  <img src="docs/img/readme-banner.webp" width="440" alt="mojimask">
</p>

<p align="center">
  Face masking for dance parties, protests and other complex videos.<br>
  Offline, free and open source.
</p>

<p align="center">
  <a href="https://github.com/acgourley/mojimask/releases/latest/download/Mojimask.dmg"><b>Download for macOS</b></a>
  &nbsp;·&nbsp; <a href="docs/media/jazz-walkthrough.mp4">Watch the 45 second demo</a>
  &nbsp;·&nbsp; <a href="https://acgourley.github.io/mojimask/">Website</a>
  <br><sub>Requires an Apple-silicon Mac with macOS 15 or later. Nothing leaves your Mac.</sub>
</p>

<p align="center">
  <img src="docs/img/disguise-masquerave.webp" width="520" alt="A tarantella dance scene, every face covered by a smiling emoji">
</p>

## What it does

Crowds, strobe lights, fast motion can confuse other tools. Mojimask uses research grade machine learning to find,
follow and understand each person in the scene.

- **Finding:** YOLOv5-Face and RetinaFace detect and double-check faces. YOLO11x-pose finds heads even when faces
  are hidden, and tilts the masks.
- **Following:** optical flow follows each person from frame to frame, and SFace and OSNet-AIN recognise them when
  they reappear.
- **Depth:** Depth Anything V2 keeps faces at different distances apart and layers covers behind whatever is in
  front of them.
- **Human judgement:** for the perfectionist, the app will still ask you to weigh in on the tricky cases. Automatic
  judgements can be modified; set custom masks or unmask the consenting.
- **Masks:** emoji presets, blur, pixelate, or even upload a sticker, with options to match scene lighting and body
  pose.

## Building it

mojimask is a Rust app with a Qt Quick window (in `native/`). Building it needs an Apple-silicon Mac with
macOS 15, Xcode, [rustup](https://rustup.rs) (it fetches the Rust version in `native/rust-toolchain.toml`) and
Homebrew's Qt, GStreamer, cmake and ninja:

```bash
brew install qt gstreamer cmake ninja
# put the model files in models/: see models/README.md
native/build.sh                                               # build it
native/app/build/Mojimask.app/Contents/MacOS/Mojimask --dev   # run it from source
native/packaging/make_app.sh --dmg                            # or package it: native/dist/Mojimask.app and .dmg
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), including how to work with AI coding tools here. How to run, change and
measure the app: [AGENTS.md](AGENTS.md), written for coding agents and people alike.

## License

AGPL-3.0 (see `LICENSE`), which the bundled YOLO11 pose model (Ultralytics, AGPL-3.0) requires. Model
weights have their own licenses, listed in `models/README.md`; everything else bundled is in
`THIRD_PARTY_NOTICES.md`.
