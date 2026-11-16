"""Draw the app icon with AppKit and pack it as .icns with iconutil: The Laughing Cavalier (Hals, 1624; public
domain; web/assets/icon_art.jpg) on the macOS icon tile, his face covered by the disguise emoji, drawn by this Mac's own
emoji font the way the app draws its covers. Sizes follow the design: a 1024 canvas, an 824 tile with a 185 corner
radius and a soft shadow, the emoji 402 across, turned 11 degrees.

    .venv/bin/python packaging/make_icon.py build/AppIcon.icns
"""
import os
import shutil
import subprocess
import sys

from AppKit import (NSAffineTransform, NSBezierPath, NSBitmapImageRep, NSColor, NSCompositingOperationSourceOver, NSFont,
                    NSFontAttributeName, NSGraphicsContext, NSImage, NSMakeRect, NSPNGFileType, NSShadow, NSString)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "web", "assets", "icon_art.jpg")
EMOJI = "🥸"
TILE, RADIUS = 824 / 1024, 185.4 / 1024        # of the canvas
EMOJI_AT, EMOJI_SIZE, EMOJI_TURN = (165 / 824, 189 / 824), 402 / 824, 11.17  # of the tile: top-left, width; degrees anticlockwise


def draw(px, path):
    rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, px, px, 8, 4, True, False, "NSCalibratedRGBColorSpace", 0, 0)
    NSGraphicsContext.saveGraphicsState()
    ctx = NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep)
    NSGraphicsContext.setCurrentContext_(ctx)
    ctx.setImageInterpolation_(3)  # high
    t = px * TILE
    inset = (px - t) / 2
    tile = NSMakeRect(inset, inset, t, t)
    shape = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(tile, px * RADIUS, px * RADIUS)
    # the tile's shadow, then the painting clipped to the tile
    NSGraphicsContext.saveGraphicsState()
    sh = NSShadow.alloc().init()
    sh.setShadowColor_(NSColor.colorWithCalibratedWhite_alpha_(0, 0.35)); sh.setShadowOffset_((0, -px * 12 / 1024)); sh.setShadowBlurRadius_(px * 28 / 1024)
    sh.set()
    NSColor.blackColor().setFill(); shape.fill()
    NSGraphicsContext.restoreGraphicsState()
    NSGraphicsContext.saveGraphicsState()
    shape.addClip()
    NSImage.alloc().initWithContentsOfFile_(ART).drawInRect_fromRect_operation_fraction_(tile, NSMakeRect(0, 0, 0, 0), NSCompositingOperationSourceOver, 1.0)
    # the emoji: its box's top-left corner at EMOJI_AT (y down, as in the design), turned about that corner
    size = t * EMOJI_SIZE
    x, y_top = inset + t * EMOJI_AT[0], inset + t - t * EMOJI_AT[1]
    tf = NSAffineTransform.transform()
    tf.translateXBy_yBy_(x, y_top); tf.rotateByDegrees_(EMOJI_TURN); tf.concat()
    s = NSString.stringWithString_(EMOJI)
    attrs = {NSFontAttributeName: NSFont.systemFontOfSize_(size * 1.1)}  # (the glyph is narrower than its point size)
    sz = s.sizeWithAttributes_(attrs)
    s.drawAtPoint_withAttributes_(((size - sz.width) / 2, -size + (size - sz.height) / 2), attrs)
    NSGraphicsContext.restoreGraphicsState()
    NSGraphicsContext.restoreGraphicsState()
    rep.representationUsingType_properties_(NSPNGFileType, {}).writeToFile_atomically_(path, True)


def main(out):
    iconset = os.path.splitext(out)[0] + ".iconset"
    shutil.rmtree(iconset, ignore_errors=True)
    os.makedirs(iconset)
    for base in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            name = f"icon_{base}x{base}{'@2x' if scale == 2 else ''}.png"
            draw(base * scale, os.path.join(iconset, name))
    subprocess.run(["iconutil", "-c", "icns", iconset, "-o", out], check=True)
    shutil.rmtree(iconset)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build/AppIcon.icns")
