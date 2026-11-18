// macOS bits: the window's title bar becomes part of the page (as in the current app: the bar is the title bar), dark
// appearance, and the window's background in the theme's colour so a resize never flashes white.
#import <AppKit/AppKit.h>
#include <QWindow>

void macStyleWindow(QWindow *w) {
    NSView *v = (__bridge NSView *)reinterpret_cast<void *>(w->winId());
    NSWindow *nw = v.window;
    if (!nw) return;
    NSApp.appearance = [NSAppearance appearanceNamed:NSAppearanceNameDarkAqua];
    nw.titlebarAppearsTransparent = YES;
    nw.titleVisibility = NSWindowTitleHidden;
    nw.styleMask |= NSWindowStyleMaskFullSizeContentView;
    nw.backgroundColor = [NSColor colorWithSRGBRed:0x0D / 255.0 green:0x0B / 255.0 blue:0x17 / 255.0 alpha:1];
    nw.movableByWindowBackground = NO;
}
