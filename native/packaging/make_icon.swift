// Draw the app icon with AppKit and pack it as .icns with iconutil: The Laughing Cavalier (Hals, 1624; public domain;
// native/web/assets/icon_art.jpg) on the macOS icon tile, his face covered by the disguise emoji, drawn by this Mac's
// own emoji font the way the app draws its covers. Sizes follow the design: a 1024 canvas, an 824 tile with a 185
// corner radius and a soft shadow, the emoji 402 across, turned 11 degrees. Drawn at build time, so neither the
// picture nor Apple's emoji is kept in git. (Port of the Python app's packaging/make_icon.py, line for line.)
//
//     swift native/packaging/make_icon.swift ART OUT.icns
import AppKit

let args = CommandLine.arguments
guard args.count == 3 else {
    FileHandle.standardError.write("usage: make_icon.swift ART OUT.icns\n".data(using: .utf8)!)
    exit(2)
}
let art = args[1], out = args[2]
guard let painting = NSImage(contentsOfFile: art) else {
    FileHandle.standardError.write("make_icon: can't read \(art)\n".data(using: .utf8)!)
    exit(1)
}
let emoji = "🥸"
let tileF = 824.0 / 1024, radiusF = 185.4 / 1024                    // of the canvas
let emojiAt = (165.0 / 824, 189.0 / 824), emojiSize = 402.0 / 824, emojiTurn = 11.17  // of the tile; degrees anticlockwise

func draw(_ px: Int, _ path: String) {
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: px, pixelsHigh: px, bitsPerSample: 8, samplesPerPixel: 4,
                               hasAlpha: true, isPlanar: false, colorSpaceName: .calibratedRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    let ctx = NSGraphicsContext(bitmapImageRep: rep)!
    NSGraphicsContext.current = ctx
    ctx.imageInterpolation = .high
    let p = Double(px), t = p * tileF, inset = (p - t) / 2
    let tile = NSRect(x: inset, y: inset, width: t, height: t)
    let shape = NSBezierPath(roundedRect: tile, xRadius: p * radiusF, yRadius: p * radiusF)
    // the tile's shadow, then the painting clipped to the tile
    NSGraphicsContext.saveGraphicsState()
    let sh = NSShadow()
    sh.shadowColor = NSColor(calibratedWhite: 0, alpha: 0.35)
    sh.shadowOffset = NSSize(width: 0, height: -p * 12 / 1024)
    sh.shadowBlurRadius = p * 28 / 1024
    sh.set()
    NSColor.black.setFill()
    shape.fill()
    NSGraphicsContext.restoreGraphicsState()
    NSGraphicsContext.saveGraphicsState()
    shape.addClip()
    painting.draw(in: tile, from: .zero, operation: .sourceOver, fraction: 1.0)
    // the emoji: its box's top-left corner at emojiAt (y down, as in the design), turned about that corner
    let size = t * emojiSize
    let x = inset + t * emojiAt.0, yTop = inset + t - t * emojiAt.1
    let tf = NSAffineTransform()
    tf.translateX(by: x, yBy: yTop)
    tf.rotate(byDegrees: emojiTurn)
    tf.concat()
    let attrs: [NSAttributedString.Key: Any] = [.font: NSFont.systemFont(ofSize: size * 1.1)]  // (the glyph is narrower than its point size)
    let s = emoji as NSString
    let sz = s.size(withAttributes: attrs)
    s.draw(at: NSPoint(x: (size - sz.width) / 2, y: -size + (size - sz.height) / 2), withAttributes: attrs)
    NSGraphicsContext.restoreGraphicsState()
    NSGraphicsContext.restoreGraphicsState()
    try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: path), options: .atomic)
}

let iconset = (out as NSString).deletingPathExtension + ".iconset"
let fm = FileManager.default
try? fm.removeItem(atPath: iconset)
try! fm.createDirectory(atPath: iconset, withIntermediateDirectories: true)
for base in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        draw(base * scale, "\(iconset)/icon_\(base)x\(base)\(scale == 2 ? "@2x" : "").png")
    }
}
let p = Process()
p.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
p.arguments = ["-c", "icns", iconset, "-o", out]
try! p.run()
p.waitUntilExit()
try? fm.removeItem(atPath: iconset)
if p.terminationStatus != 0 { exit(1) }
print(out)
