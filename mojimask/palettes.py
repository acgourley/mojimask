"""Emoji sets ("palettes") people can pick in the review tool. Definitions live in palettes.json,
shared with the page (emoji_set_ui.js). Choice is stored in edits:
    edits["palette"]        = a preset key from palettes.json, or "custom"
    edits["palette_custom"] = list of emoji (when "custom"; only Custom is editable)
An unknown key (an old project's retired preset) falls back to DEFAULT.
Assignment hash must match review.html emojiFor(): same person -> same slot in whatever set is active.
"""
import json
import os

from mojimask.paths import WEB

with open(os.path.join(WEB, "palettes.json"), encoding="utf-8") as _fh:
    PRESETS = {p["key"]: p for p in json.load(_fh)}
# the preset marked "default": true in palettes.json (the pages and facts.py read the same flag)
DEFAULT = next((k for k, p in PRESETS.items() if p.get("default")), next(iter(PRESETS)))


def for_edits(edits):
    key = (edits or {}).get("palette") or DEFAULT
    if key == "custom" and edits.get("palette_custom"):
        return list(edits["palette_custom"])
    return PRESETS.get(key, PRESETS[DEFAULT])["emoji"]


def pick(tid, pal):
    h = sum((i + 1) * ord(ch) for i, ch in enumerate(str(tid).removeprefix("@")))
    return pal[h % len(pal)]


def assignment(per_frame, edits):
    """{person tid: emoji} for every person in the resolved frames, honouring per-person overrides.
    Passed to track.render as its `overrides`, so the renderer needs no palette knowledge."""
    pal = for_edits(edits)
    out = {tid: pick(tid, pal) for f in per_frame for tid, *_ in f}
    out.update((edits or {}).get("emoji") or {})
    return out
