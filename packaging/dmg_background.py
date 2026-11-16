"""The picture behind the disk image's window (make_dmg.sh): drag the app onto Applications, then how to open it the
first time, since the app isn't notarized (no paid Apple developer account). Drawn at build time, at 1x and 2x, so no
picture is kept in git.

    python packaging/dmg_background.py OUT_DIR NAME   -> OUT_DIR/background.png, OUT_DIR/background@2x.png

The layout (window size, where the two icons sit) is shared with make_dmg.sh through the numbers printed last.
Finder writes the icons' names itself, white in dark mode and black in light mode, so the band they sit on is a
middle tone that both can be read on.
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

N = sys.argv[2] if len(sys.argv) > 2 else "Mojimask"  # the app's name as Finder shows it under the icon

W, H = 640, 470                 # the window's content, in points
APP_AT, APPS_AT = (170, 150), (470, 150)   # icon centres (Finder's positions)
ICON = 112
BG, BAND, TEXT, SOFT, DIM, ACCENT = "#0D0B17", "#7A72A0", "#FFFFFF", "#C9C4DB", "#9A93B0", "#FFD23F"
ROUNDED = "/System/Library/Fonts/SFNSRounded.ttf"
PLAIN = "/System/Library/Fonts/SFNS.ttf"

STEPS = [
    (f"Open {N} from Applications.",
     "macOS says it can't verify it. Click Done."),
    ("Open System Settings › Privacy & Security.",
     "Scroll down to Security and click Open Anyway."),
    ("Click Open Anyway again and confirm.",
     "Only once. After that it opens like any app."),
]


def font(path, size, weight):
    f = ImageFont.truetype(path, size)
    try:
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f


def draw(s):
    im = Image.new("RGB", (W * s, H * s), BG)
    d = ImageDraw.Draw(im)
    S = lambda *v: [x * s for x in v]

    # the icons' row: a band under both, for Finder's names
    d.rounded_rectangle(S(40, 76, W - 40, 236), radius=18 * s, fill="#1A1628")
    for cx, cy in (APP_AT, APPS_AT):
        d.rounded_rectangle(S(cx - 62, cy + 62, cx + 62, cy + 84), radius=11 * s, fill=BAND)
    # the arrow
    y = APP_AT[1]
    d.line(S(APP_AT[0] + 78, y, APPS_AT[0] - 90, y), fill=ACCENT, width=5 * s)
    tip = APPS_AT[0] - 78
    d.polygon(S(tip, y, tip - 18, y - 13, tip - 18, y + 13), fill=ACCENT)

    title = font(ROUNDED, 22 * s, "Bold")
    d.text((W * s // 2, 40 * s), f"Drag {N} into Applications", font=title, fill=TEXT, anchor="mm")

    head = font(ROUNDED, 15 * s, "Bold")
    bold = font(PLAIN, 13 * s, "Semibold")
    body = font(PLAIN, 12 * s, "Regular")
    small = font(PLAIN, 11 * s, "Regular")
    x0, y0 = 56, 262
    d.text((x0 * s, y0 * s), "The first time you open it", font=head, fill=TEXT)
    for i, (a, b) in enumerate(STEPS):
        ty = y0 + 32 + i * 44
        d.ellipse(S(x0, ty, x0 + 22, ty + 22), fill=ACCENT)
        d.text(((x0 + 11) * s, (ty + 11) * s), str(i + 1), font=font(ROUNDED, 13 * s, "Bold"), fill="#1A1405", anchor="mm")
        d.text(((x0 + 34) * s, (ty - 1) * s), a, font=bold, fill=TEXT)
        d.text(((x0 + 34) * s, (ty + 17) * s), b, font=body, fill=SOFT)
    note = (f"{N} is free and open source, and isn't signed through Apple's paid developer program,\n"
            "so macOS asks once. On macOS 13 or 14 you can instead Control-click it and choose Open.")
    d.multiline_text((x0 * s, (H - 46) * s), note, font=small, fill=DIM, spacing=3 * s)
    return im


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    draw(1).save(os.path.join(out, "background.png"))
    draw(2).save(os.path.join(out, "background@2x.png"))
    print(W, H, *APP_AT, *APPS_AT, ICON)
