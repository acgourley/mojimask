// The picture behind the disk image's window (make_dmg.sh): drag the app onto Applications, then how to open it the
// first time, since the app is signed ad hoc, not notarized. Drawn at build time with the Mac's own fonts, at 1x and
// 2x, so no picture is kept in git. (Port of the Python app's packaging/dmg_background.py: the same layout, colours
// and words, drawn with AppKit instead of Pillow; the native app needs macOS 15, so the old Control-click tip is gone.)
//
//     swift native/packaging/dmg_background.swift OUT_DIR NAME   -> OUT_DIR/background.png, background@2x.png
//
// The layout (window size, where the two icons sit) is shared with make_dmg.sh through the numbers printed last.
// Finder writes the icons' names itself, white in dark mode and black in light mode, so the band they sit on is a
// middle tone that both can be read on.
import AppKit

let args = CommandLine.arguments
guard args.count >= 2 else {
    FileHandle.standardError.write("usage: dmg_background.swift OUT_DIR [NAME]\n".data(using: .utf8)!)
    exit(2)
}
let outDir = args[1]
let N = args.count > 2 ? args[2] : "Mojimask"  // the app's name as Finder shows it under the icon

let W = 640.0, H = 470.0                  // the window's content, in points
let appAt = (170.0, 150.0), appsAt = (470.0, 150.0)   // icon centres (Finder's positions)
let ICON = 112
func hex(_ s: String) -> NSColor {
    let v = Int(s.dropFirst(), radix: 16)!
    return NSColor(srgbRed: CGFloat((v >> 16) & 255) / 255, green: CGFloat((v >> 8) & 255) / 255, blue: CGFloat(v & 255) / 255, alpha: 1)
}
let BG = hex("#0D0B17"), ROW = hex("#1A1628"), BAND = hex("#7A72A0"), TEXT = hex("#FFFFFF"), SOFT = hex("#C9C4DB"),
    DIM = hex("#9A93B0"), ACCENT = hex("#FFD23F"), ON_ACCENT = hex("#1A1405")

let steps = [
    ("Open \(N) from Applications.", "macOS says it can't verify it. Click Done."),
    ("Open System Settings › Privacy & Security.", "Scroll down to Security and click Open Anyway."),
    ("Click Open Anyway again and confirm.", "Only once. After that it opens like any app."),
]

func font(_ size: Double, _ weight: NSFont.Weight, rounded: Bool = false) -> NSFont {
    let f = NSFont.systemFont(ofSize: size, weight: weight)
    guard rounded, let d = f.fontDescriptor.withDesign(.rounded) else { return f }
    return NSFont(descriptor: d, size: size) ?? f
}

/// Text with its top-left at (x, y), or centred on (x, y) when `centred`; y down.
func text(_ s: String, _ x: Double, _ y: Double, _ f: NSFont, _ c: NSColor, centred: Bool = false) {
    let attrs: [NSAttributedString.Key: Any] = [.font: f, .foregroundColor: c]
    let a = NSAttributedString(string: s, attributes: attrs)
    let sz = a.size()
    a.draw(at: centred ? NSPoint(x: x - sz.width / 2, y: y - sz.height / 2) : NSPoint(x: x, y: y))
}

func draw(_ s: Int, _ path: String) {
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(W) * s, pixelsHigh: Int(H) * s, bitsPerSample: 8,
                               samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                               bytesPerRow: 0, bitsPerPixel: 0)!
    rep.size = NSSize(width: W, height: H)  // points; the rep holds s pixels per point
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    // y down, as the design is laid out
    let flip = NSAffineTransform()
    flip.translateX(by: 0, yBy: H)
    flip.scaleX(by: 1, yBy: -1)
    flip.concat()
    BG.setFill()
    NSRect(x: 0, y: 0, width: W, height: H).fill()

    // the icons' row: a band under both, for Finder's names
    ROW.setFill()
    NSBezierPath(roundedRect: NSRect(x: 40, y: 76, width: W - 80, height: 160), xRadius: 18, yRadius: 18).fill()
    BAND.setFill()
    for (cx, cy) in [appAt, appsAt] {
        NSBezierPath(roundedRect: NSRect(x: cx - 62, y: cy + 62, width: 124, height: 22), xRadius: 11, yRadius: 11).fill()
    }
    // the arrow
    let y = appAt.1, tip = appsAt.0 - 78
    ACCENT.set()
    let line = NSBezierPath()
    line.move(to: NSPoint(x: appAt.0 + 78, y: y))
    line.line(to: NSPoint(x: appsAt.0 - 90, y: y))
    line.lineWidth = 5
    line.stroke()
    let head = NSBezierPath()
    head.move(to: NSPoint(x: tip, y: y))
    head.line(to: NSPoint(x: tip - 18, y: y - 13))
    head.line(to: NSPoint(x: tip - 18, y: y + 13))
    head.close()
    head.fill()

    // the flipped context draws text upside down unless the text knows; NSAttributedString.draw honours a flipped
    // NSGraphicsContext only when it is created flipped, so text goes through a flipped context of its own
    NSGraphicsContext.restoreGraphicsState()
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(cgContext: NSGraphicsContext(bitmapImageRep: rep)!.cgContext, flipped: true)
    let ctm = NSAffineTransform()
    ctm.translateX(by: 0, yBy: H)
    ctm.scaleX(by: 1, yBy: -1)
    ctm.concat()

    text("Drag \(N) into Applications", W / 2, 40, font(22, .bold, rounded: true), TEXT, centred: true)
    let x0 = 56.0, y0 = 262.0
    text("The first time you open it", x0, y0, font(15, .bold, rounded: true), TEXT)
    for (i, (a, b)) in steps.enumerated() {
        let ty = y0 + 32 + Double(i) * 44
        ACCENT.setFill()
        NSBezierPath(ovalIn: NSRect(x: x0, y: ty, width: 22, height: 22)).fill()
        text("\(i + 1)", x0 + 11, ty + 11, font(13, .bold, rounded: true), ON_ACCENT, centred: true)
        text(a, x0 + 34, ty - 1, font(13, .semibold), TEXT)
        text(b, x0 + 34, ty + 17, font(12, .regular), SOFT)
    }
    text("\(N) is free and open source, and isn't signed through Apple's paid developer program,", x0, H - 46, font(11, .regular), DIM)
    text("so macOS asks once.", x0, H - 46 + 14 + 3, font(11, .regular), DIM)
    NSGraphicsContext.restoreGraphicsState()
    try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: path), options: .atomic)
}

try! FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
draw(1, "\(outDir)/background.png")
draw(2, "\(outDir)/background@2x.png")
print(Int(W), Int(H), Int(appAt.0), Int(appAt.1), Int(appsAt.0), Int(appsAt.1), ICON)
