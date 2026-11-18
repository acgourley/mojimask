#!/bin/bash
# The disk image to send, as packaging/make_dmg.sh makes the Python app's: Mojimask.app beside an Applications
# shortcut, over a picture that says to drag one onto the other and how to open it the first time (dmg_background.swift
# draws it; the app is signed ad hoc, not notarized). No read-me: the picture is the instructions.
#
#   native/packaging/make_dmg.sh [native/dist/Mojimask.app]    -> native/dist/Mojimask.dmg
#
# Only macOS's own tools (hdiutil, Finder through osascript): the window's layout is set by Finder on a writable copy,
# then the image is compressed. The first run may ask to let the terminal control Finder: allow it, or the window
# opens with Finder's default layout (everything still works, just without the picture).
#
# For distribution with a Developer ID (make_app.sh's MOJIMASK_SIGN_ID): the image is signed too, and notarized and
# stapled when MOJIMASK_NOTARY_PROFILE names an `xcrun notarytool store-credentials` profile.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
APP="$(cd "$(dirname "${1:-$HERE/../dist/Mojimask.app}")" && pwd)/$(basename "${1:-Mojimask.app}")"
NAME="$(basename "$APP" .app)"
VOL="$NAME"
OUT="$(dirname "$APP")/$NAME.dmg"
[ -d "$APP" ] || { echo "No $APP: run make_app.sh first" >&2; exit 1; }
[ ! -d "/Volumes/$VOL" ] || { echo "/Volumes/$VOL is already mounted: eject it first" >&2; exit 1; }

WORK="$(mktemp -d)"
trap 'hdiutil detach -quiet "/Volumes/$VOL" 2>/dev/null || true; rm -rf "$WORK"' EXIT
STAGE="$WORK/stage"
mkdir -p "$STAGE/.background"

# the picture: 1x and 2x in one TIFF, so it's sharp on Retina screens
read -r W H AX AY BX BY ICON < <(swift "$HERE/dmg_background.swift" "$WORK/bg" "$NAME")
tiffutil -cathidpicheck "$WORK/bg/background.png" "$WORK/bg/background@2x.png" -out "$STAGE/.background/background.tiff" 2>/dev/null

ditto "$APP" "$STAGE/$NAME.app"
ln -s /Applications "$STAGE/Applications"

SIZE_MB=$(( $(du -sm "$STAGE" | cut -f1) + 60 ))
hdiutil create -quiet -srcfolder "$STAGE" -volname "$VOL" -fs HFS+ -format UDRW -size "${SIZE_MB}m" "$WORK/rw.dmg"
hdiutil attach -quiet -readwrite -noverify -noautoopen "$WORK/rw.dmg"

# Finder writes the window's layout into the volume's .DS_Store
osascript <<EOF || echo "Couldn't arrange the window through Finder (see the note at the top); continuing." >&2
tell application "Finder"
  repeat 40 times  -- (Finder sees a just-attached disk a moment later)
    if exists disk "$VOL" then exit repeat
    delay 0.25
  end repeat
  tell disk "$VOL"
    open
    set current view of container window to icon view
    set toolbar visible of container window to false
    set statusbar visible of container window to false
    set the bounds of container window to {200, 120, 200 + $W, 120 + $H + 28}  -- (+ the title bar)
    set opts to the icon view options of container window
    set arrangement of opts to not arranged
    set icon size of opts to $ICON
    set text size of opts to 13
    set background picture of opts to file ".background:background.tiff"
    set position of item "$NAME.app" of container window to {$AX, $AY}
    set position of item "Applications" of container window to {$BX, $BY}
    set extension hidden of item "$NAME.app" of container window to true
    close
    open
    update without registering applications
    delay 1
    close
  end tell
end tell
EOF
chmod -Rf go-w "/Volumes/$VOL" 2>/dev/null || true
sync
hdiutil detach -quiet "/Volumes/$VOL"

rm -f "$OUT"
hdiutil convert -quiet "$WORK/rw.dmg" -format ULFO -o "$OUT"
hdiutil verify -quiet "$OUT"
if [ -n "${MOJIMASK_SIGN_ID:-}" ]; then
  codesign --force --sign "$MOJIMASK_SIGN_ID" --timestamp "$OUT"
  if [ -n "${MOJIMASK_NOTARY_PROFILE:-}" ]; then
    xcrun notarytool submit "$OUT" --keychain-profile "$MOJIMASK_NOTARY_PROFILE" --wait
    xcrun stapler staple "$OUT"
    spctl --assess --type open --context context:primary-signature -v "$OUT"
  fi
fi
du -sh "$OUT"
