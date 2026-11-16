"""What the app says about itself, in one place: its name and where its source lives. Used for the metadata written
into every export (video_io: the app's name and this address, nothing else), the window and menu titles (desktop.py),
the splash (splash.py) and the export's corner mark (track.py).

The name and the wordmark's look are not set here but in brand.js, which the pages load too: its BRAND object is
plain JSON, read below, so the pages and the Python side can't disagree. Change the name's case there."""
import json
import os

from mojimask.paths import WEB as _HERE  # brand.js and fonts/


def _brand():
    with open(os.path.join(_HERE, "brand.js"), encoding="utf-8") as f:
        src = f.read()
    return json.JSONDecoder().raw_decode(src, src.index("{", src.index("const BRAND")))[0]


BRAND = _brand()  # name, split, font, weight, tracking (em), first, second: see brand.js
NAME = BRAND["name"]
WORDMARK = (NAME[:BRAND["split"]], NAME[BRAND["split"]:])  # the two parts, e.g. ("Moji", "Mask")
FONT_FILE = os.path.join(_HERE, "fonts", BRAND["font"] + ".woff2")
URL = "https://github.com/acgourley/mojimask"
MADE_WITH = f"Made with {NAME}: {URL}"  # an export's description
