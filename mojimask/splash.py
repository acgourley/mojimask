"""The Mac app's opening page: the app icon and the wordmark on black, while the engine starts behind it (well under a
second; the models load at the first scan, so there is nothing to show progress for). When the app's page is ready,
desktop.py fades this to black (FADE_S) and swaps the page in.

One page of HTML with everything inside it (the icon's painting and the fonts as data: URLs), because it is shown
before the app's own server is up. The icon is drawn the way packaging/make_icon.py draws the real one: The Laughing
Cavalier (assets/icon_art.jpg) on a rounded tile, the disguise emoji over his face, in the Mac's own emoji font.

    html()   (the app server shows it at /splash in developer mode)
"""
import base64
import html as _html
import os

from mojimask import about

from mojimask.paths import WEB as HERE
FADE_S = 0.3
# packaging/make_icon.py's numbers, as fractions of the tile
RADIUS, EMOJI_AT, EMOJI_SIZE, EMOJI_TURN = 185.4 / 824, (165 / 824, 189 / 824), 402 / 824, 11.17


def _data(path, kind):
    try:
        return f"data:{kind};base64," + base64.b64encode(open(os.path.join(HERE, path), "rb").read()).decode()
    except OSError:
        return None


def html():
    faces = ""  # (not bundled: the Mac's own rounded font stands in)
    B = about.BRAND  # the wordmark's look and the name's case (brand.js), as theme.css's .wordmark.big draws it
    font = _data(os.path.relpath(about.FONT_FILE, HERE), "font/woff2")
    if font:
        faces = f'@font-face {{ font-family: "{B["font"]}"; font-weight: 200 1000; src: url({font}) format("woff2"); }}'
    art = _data("assets/icon_art.jpg", "image/jpeg")
    icon = (f'<div class="icon" style="background-image:url({art})"><span>🥸</span></div>' if art else "")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{_html.escape(about.NAME)}</title>
<style>
  {faces}
  html, body {{ margin: 0; height: 100%; overflow: hidden; background: #000; color-scheme: dark; }}
  body {{ display: grid; place-items: center; -webkit-user-select: none; user-select: none; cursor: default; }}
  main {{ display: flex; flex-direction: column; align-items: center; gap: 28px; transition: opacity {FADE_S}s ease-in; }}
  body.out main {{ opacity: 0; }}
  .icon {{ position: relative; width: 168px; aspect-ratio: 1; border-radius: {RADIUS * 100:.2f}%; overflow: hidden;
           background: center / cover; box-shadow: 0 6px 24px #0008; container-type: inline-size; }}
  .icon span {{ position: absolute; left: {EMOJI_AT[0] * 100:.2f}%; top: {EMOJI_AT[1] * 100:.2f}%;
                width: {EMOJI_SIZE * 100:.2f}cqw; height: {EMOJI_SIZE * 100:.2f}cqw; display: grid; place-items: center;
                font-size: {EMOJI_SIZE * 110:.2f}cqw; line-height: 1;
                transform: rotate({-EMOJI_TURN}deg); transform-origin: top left; }}
  .word {{ font: {B["weight"]} 56px/1 "{B["font"]}", ui-rounded, "SF Pro Rounded", -apple-system, sans-serif;
           letter-spacing: {B["tracking"]}em; font-kerning: normal; color: {B["first"]}; }}
  .word b {{ color: {B["second"]}; font-weight: inherit; }}
</style></head>
<body class="pywebview-drag-region">
  <main>{icon}<div class="word" aria-label="{_html.escape(about.NAME)}">{_html.escape(about.WORDMARK[0])}<b>{_html.escape(about.WORDMARK[1])}</b></div></main>
</body></html>"""
