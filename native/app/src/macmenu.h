// The Mac menu bar's few actions that QML can't do itself (platform_mac.mm): the standard About panel (the bundle's
// name, version and copyright, from Info.plist) and Window > Bring All to Front. Main.qml's MenuBar calls them as `mac`.
// tidy() tells macOS which menus are the Window and Help menus (Qt doesn't), so it adds the window list and its tiling
// items to one and the search field to the other, and drops the app menu's doubled separator (the hidden Settings slot).
// menus() is the menu bar as macOS has it (titles, keys, enabled) and press() picks an item as a click does, for
// checking them: `mmctl eval 'mac.menus()'`, `mmctl eval 'mac.press("Edit", "Paste")'`.
#pragma once
#include <QObject>
#include <QVariant>

class MacMenu : public QObject {
    Q_OBJECT
public:
    using QObject::QObject;
    Q_INVOKABLE void about();
    Q_INVOKABLE void bringAllToFront();
    Q_INVOKABLE void tidy(const QString &windowMenu, const QString &helpMenu);
    Q_INVOKABLE QVariantList menus() const;
    Q_INVOKABLE bool press(const QString &menu, const QString &item);
};
