// macOS bits: the window's title bar becomes part of the page (as in the current app: the bar is the title bar), dark
// appearance, and the window's background in the theme's colour so a resize never flashes white.
#import <AppKit/AppKit.h>
#import <objc/runtime.h>
#include <QGuiApplication>
#include <QMetaObject>
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

// the menu bar's About and Bring All to Front (macmenu.h)
#include "macmenu.h"

void MacMenu::about() { [NSApp orderFrontStandardAboutPanel:nil]; }
void MacMenu::bringAllToFront() { [NSApp arrangeInFront:nil]; }

void MacMenu::tidy(const QString &windowMenu, const QString &helpMenu) {
    NSMenu *bar = NSApp.mainMenu;
    if (NSMenu *w = [bar itemWithTitle:windowMenu.toNSString()].submenu) NSApp.windowsMenu = w;
    if (NSMenu *h = [bar itemWithTitle:helpMenu.toNSString()].submenu) NSApp.helpMenu = h;
    // the app menu: a separator straight after another (hidden items between don't count) is hidden
    NSMenu *app = bar.numberOfItems ? [bar itemAtIndex:0].submenu : nil;
    bool lastSep = true;
    for (NSMenuItem *i in app.itemArray) {
        if (i.isSeparatorItem) { i.hidden = lastSep; lastSep = true; }
        else if (!i.isHidden) lastSep = false;
    }
}

static QVariantList describeMenu(NSMenu *m) {
    QVariantList out;
    for (NSMenuItem *i in m.itemArray) {
        if (i.isHidden) continue;
        if (i.isSeparatorItem) { out << QVariantMap{{"separator", true}}; continue; }
        QString mods;
        NSEventModifierFlags f = i.keyEquivalentModifierMask;
        if (i.keyEquivalent.length) {
            if (f & NSEventModifierFlagControl) mods += "⌃";
            if (f & NSEventModifierFlagOption) mods += "⌥";
            if (f & NSEventModifierFlagShift) mods += "⇧";
            if (f & NSEventModifierFlagCommand) mods += "⌘";
        }
        // enabled as the open menu shows it: an auto-enabling menu asks the item's target (validateMenuItem:)
        bool on = i.isEnabled;
        id t = i.target ?: [NSApp targetForAction:i.action];
        if (m.autoenablesItems && i.action)
            on = t && (![t respondsToSelector:@selector(validateMenuItem:)] || [t validateMenuItem:i]);
        QVariantMap e{{"title", QString::fromNSString(i.title)}, {"enabled", on}};
        if (i.keyEquivalent.length) e["key"] = mods + QString::fromNSString(i.keyEquivalent).toUpper();
        if (i.action) e["action"] = QString::fromNSString(NSStringFromSelector(i.action));
        if (t) e["target"] = QString::fromNSString(NSStringFromClass([t class]));
        if (i.submenu) e["items"] = describeMenu(i.submenu);
        out << e;
    }
    return out;
}

QVariantList MacMenu::menus() const { return describeMenu(NSApp.mainMenu); }

bool MacMenu::press(const QString &menu, const QString &item) {
    NSMenu *m = [NSApp.mainMenu itemWithTitle:menu.toNSString()].submenu;
    [m update];   // enable its items as opening it would
    NSInteger i = [m indexOfItemWithTitle:item.toNSString()];
    if (i < 0 || ![m itemAtIndex:i].isEnabled) return false;
    [m performActionForItemAtIndex:i];
    return true;
}

// Edit > Cut, Copy, Paste, Select All: Qt gives these items AppKit's own actions (cut: and the rest, by their titles),
// which go to the window's first responder, Qt's view, and it has no such methods: the items stayed grey and a click
// did nothing. Give the view those methods: each calls the focused text field's own (QML TextInput / TextEdit), and the
// menu shows them only when the field can (something selected to cut or copy, not read-only to cut or paste).
static QObject *editTarget(SEL a) {
    QObject *f = QGuiApplication::focusObject();
    if (!f || f->metaObject()->indexOfMethod("selectAll()") < 0) return nullptr;
    const bool sel = !f->property("selectedText").toString().isEmpty(), ro = f->property("readOnly").toBool();
    if (a == @selector(cut:)) return sel && !ro ? f : nullptr;
    if (a == @selector(copy:)) return sel ? f : nullptr;
    if (a == @selector(paste:)) return !ro ? f : nullptr;
    return f;
}

void macEditActions() {
    Class view = NSClassFromString(@"QNSView");
    if (!view) return;
    for (NSString *name in @[@"cut:", @"copy:", @"paste:", @"selectAll:"]) {
        SEL sel = NSSelectorFromString(name);
        const QByteArray method = QString::fromNSString(name).chopped(1).toUtf8();
        IMP imp = imp_implementationWithBlock(^(id, id) {
            if (QObject *f = editTarget(sel)) QMetaObject::invokeMethod(f, method.constData());
        });
        class_addMethod(view, sel, imp, "v@:@");
    }
    // the view's menu check: ours for these four, Qt's (if it has one) for the rest
    SEL v = @selector(validateMenuItem:);
    Method had = class_getInstanceMethod(view, v);
    IMP old = had ? method_getImplementation(had) : nullptr;
    IMP imp = imp_implementationWithBlock(^BOOL(id self, NSMenuItem *item) {
        SEL a = item.action;
        if (a == @selector(cut:) || a == @selector(copy:) || a == @selector(paste:) || a == @selector(selectAll:))
            return editTarget(a) != nullptr;
        return old ? ((BOOL (*)(id, SEL, NSMenuItem *))old)(self, v, item) : YES;
    });
    if (!class_addMethod(view, v, imp, "c@:@")) method_setImplementation(had, imp);
}
