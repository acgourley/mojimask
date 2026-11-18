// The window: the bar (web/app.html's titlebar: the wordmark and where you are on the left, the project's steps in
// the middle; it is also the Mac title bar) and the page shown. State and actions are the store's (store.state,
// store.call); this file only lays them out.
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Window
import QtQuick.Layouts
import "."
import "components"
import "pages"

ApplicationWindow {
    id: win
    objectName: "window"
    width: 1280
    height: 820
    minimumWidth: 900
    minimumHeight: 600
    visible: true
    color: Theme.bg
    title: store.state.project.open ? store.state.project.name + " — " + T.name : T.name

    readonly property var app: store.state.app
    readonly property var project: store.state.project
    readonly property string page: app.page

    // The Mac menu bar (the Python app's desktop.py menu_bar): macOS's own app menu (Services, Hide, Quit; Qt moves
    // "About …" there; macOS adds Writing Tools, Dictation and Emoji to Edit), then File, Edit, View, Window and Help.
    // Undo and Redo are a focused text field's when it has something to undo, otherwise the open project's history on
    // its server. A text field takes ⌘Z, ⌘C and the rest before the menu does. Check the bar with
    // `mmctl eval 'mac.menus()'` and `mac.press("Edit", "Undo")` (macmenu.h). No title may start with "Exit", "Quit",
    // "About", "Preferences" or "Settings" unless it should move to the app menu (Qt's guess at a menu item's role),
    // hence "Leave Full Screen".
    readonly property var field: activeFocusItem && typeof activeFocusItem.selectAll === "function"
                                 && activeFocusItem.selectedText !== undefined ? activeFocusItem : null
    function undoRedo(redo) {
        if (field && (redo ? field.canRedo : field.canUndo)) return redo ? field.redo() : field.undo()
        if (project.open) store.call(redo ? "redo" : "undo")
    }
    function openFile() {
        if (page === "editor" && !store.state.editor.project) return editorPage.openVideo()   // --video: the dump editor
        if (page !== "projects") store.call("go", { page: "projects" })
        projectsPage.choose()
    }
    // macOS's Window and Help menus and the app menu's separators (macmenu.h), once Qt has made the native menus: they
    // aren't there yet when the window loads, so on each activation (it's cheap and the same each time)
    onActiveChanged: if (active) mac.tidy(T.t("menu.window"), T.t("menu.help"))
    menuBar: MenuBar {
        objectName: "menubar"
        Menu {
            title: T.t("menu.file")
            Action { objectName: "menu-open"; text: T.t("menu.open"); shortcut: StandardKey.Open; onTriggered: win.openFile() }
            MenuSeparator {}
            Action { objectName: "menu-close"; text: T.t("menu.close_window"); shortcut: StandardKey.Close; onTriggered: win.close() }
        }
        Menu {
            title: T.t("menu.edit")
            Action {
                objectName: "menu-undo"; text: T.t("menu.undo"); shortcut: StandardKey.Undo
                enabled: (win.field && win.field.canUndo) || (win.project.open && win.project.can_undo)
                onTriggered: win.undoRedo(false)
            }
            Action {
                objectName: "menu-redo"; text: T.t("menu.redo"); shortcut: StandardKey.Redo
                enabled: (win.field && win.field.canRedo) || (win.project.open && win.project.can_redo)
                onTriggered: win.undoRedo(true)
            }
            MenuSeparator {}
            // these four are AppKit's cut: copy: paste: selectAll: (Qt picks them by title), handled for the focused text
            // field in platform_mac.mm, which also enables them
            Action { objectName: "menu-cut"; text: T.t("menu.cut"); shortcut: StandardKey.Cut }
            Action { objectName: "menu-copy"; text: T.t("menu.copy"); shortcut: StandardKey.Copy }
            Action { objectName: "menu-paste"; text: T.t("menu.paste"); shortcut: StandardKey.Paste }
            Action { objectName: "menu-select-all"; text: T.t("menu.select_all"); shortcut: StandardKey.SelectAll }
        }
        Menu {
            title: T.t("menu.view")
            Action {
                objectName: "menu-full-screen"
                text: T.t(win.visibility === Window.FullScreen ? "menu.full_screen_off" : "menu.full_screen")
                shortcut: StandardKey.FullScreen
                onTriggered: win.visibility === Window.FullScreen ? win.showNormal() : win.showFullScreen()
            }
        }
        Menu {
            title: T.t("menu.window")
            Action { objectName: "menu-minimize"; text: T.t("menu.minimize"); shortcut: "Ctrl+M"; onTriggered: win.showMinimized() }
            Action { objectName: "menu-zoom"; text: T.t("menu.zoom"); onTriggered: win.visibility === Window.Maximized ? win.showNormal() : win.showMaximized() }
            MenuSeparator {}
            Action { objectName: "menu-bring-all"; text: T.t("menu.bring_all"); onTriggered: mac.bringAllToFront() }
        }
        Menu {
            title: T.t("menu.help")
            Action { objectName: "menu-about"; text: T.t("menu.about"); onTriggered: mac.about() }
            Action { objectName: "menu-github"; text: T.t("menu.github"); onTriggered: Qt.openUrlExternally(T.tables.brand.url) }
            Action { objectName: "menu-licences"; text: T.t("menu.licences"); onTriggered: Qt.openUrlExternally(noticesUrl) }
        }
    }

    header: Rectangle {
        objectName: "bar"
        height: Theme.barHeight
        color: Theme.bar
        Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Theme.line }
        // the bar is the Mac title bar (platform_mac.mm), but Qt gets its clicks: hand a drag on its empty parts to the
        // system to move the window, and a double-click zooms it, as a title bar does. The bar's buttons sit above
        // these handlers and keep their clicks.
        DragHandler {
            objectName: "bar-drag"
            target: null
            grabPermissions: PointerHandler.CanTakeOverFromAnything
            // the system's move (snapping, other screens) when it takes the drag; otherwise move the window here
            property bool own: false
            property point from
            onActiveChanged: if (active) { own = !win.startSystemMove(); from = Qt.point(win.x, win.y) }
            onTranslationChanged: if (active && own) { win.x = from.x + translation.x; win.y = from.y + translation.y }
        }
        TapHandler {
            objectName: "bar-zoom"
            onDoubleTapped: win.visibility === Window.Maximized ? win.showNormal() : win.showMaximized()
        }
        // left: the wordmark, then "Projects › name" (a way back; while a scan runs, Projects alone)
        Row {
            id: left
            anchors.left: parent.left
            anchors.leftMargin: Theme.trafficLights + 8
            anchors.verticalCenter: parent.verticalCenter
            spacing: 14
            Wordmark { objectName: "wordmark"; size: 22; anchors.verticalCenter: parent.verticalCenter }
            Row {
                objectName: "crumb"
                anchors.verticalCenter: parent.verticalCenter
                spacing: 6
                leftPadding: 8
                visible: win.project.open || win.page === "scan"
                Text {
                    id: crumbProjects
                    objectName: "crumb-projects"
                    text: T.t("nav.projects")
                    font.family: Theme.body; font.pixelSize: Theme.small
                    color: crumbMouse.containsMouse ? Theme.text : Theme.dim
                    MouseArea { id: crumbMouse; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: store.call("go", { page: "projects" }) }
                    ToolTip.visible: crumbMouse.containsMouse
                    ToolTip.delay: 600
                    ToolTip.text: T.t(win.project.open ? "nav.projects_tip_open" : "nav.projects_tip_scanning")
                }
                Icon { visible: win.project.open; name: "chevron-right"; size: 14; color: Theme.faint; anchors.verticalCenter: parent.verticalCenter }
                Text {
                    objectName: "crumb-name"
                    visible: win.project.open
                    text: win.project.name || ""
                    font.family: Theme.body; font.pixelSize: Theme.small; font.weight: Font.DemiBold
                    color: Theme.text
                    elide: Text.ElideRight
                    width: Math.min(implicitWidth, Math.max(60, nav.x - left.x - 260))
                }
            }
        }
        // the middle: the open project's four steps, with what's waiting there
        SegTabs {
            id: nav
            objectName: "tab"
            anchors.centerIn: parent
            visible: win.project.open
            keys: ["decisions", "people", "editor", "export"]
            labels: [T.t("nav.decisions"), T.t("nav.people"), T.t("nav.edit"), T.t("nav.export")]
            tips: [T.t("nav.decisions_tip"), T.t("nav.people_tip"), T.t("nav.edit_tip"), T.t("nav.export_tip")]
            counts: [win.project.open_count, win.project.people_count, -1, -1]
            current: win.page
            onPicked: (key) => store.call("go", { page: key })
        }
        Text {
            objectName: "dev-tag"
            anchors.right: parent.right
            anchors.rightMargin: 20
            anchors.verticalCenter: parent.verticalCenter
            visible: win.app.dev
            text: "DEV"
            font.family: Theme.body; font.pixelSize: 10; font.weight: Font.Bold; font.letterSpacing: 1
            color: Theme.dim
            Rectangle { anchors.fill: parent; anchors.margins: -4; radius: 5; color: "transparent"; border.color: Theme.raised2 }
        }
    }

    StackLayout {
        id: pages
        anchors.fill: parent
        currentIndex: Math.max(0, win.app.pages.indexOf(win.page))
        ProjectsPage { id: projectsPage }
        ScanPage {}
        DecisionsPage {}
        PeoplePage {}
        EditorPage { id: editorPage }
        ExportPage {}
        ResultPage {}
    }

    // developer mode: what the engine is running (app.html's status bar)
    footer: Rectangle {
        objectName: "devbar"
        visible: win.app.dev && store.state.library.dev_line !== ""
        height: visible ? 32 : 0
        color: Theme.deep
        Rectangle { width: parent.width; height: 1; color: Theme.line }
        Row {
            anchors.left: parent.left
            anchors.leftMargin: 16
            anchors.verticalCenter: parent.verticalCenter
            spacing: 8
            Rectangle { width: 7; height: 7; radius: 3.5; color: Theme.good; anchors.verticalCenter: parent.verticalCenter }
            Text { text: store.state.library.dev_line; font.family: Theme.body; font.pixelSize: Theme.tiny; color: Theme.faint }
        }
        Text {
            visible: store.state.library.stale_code
            anchors.right: parent.right
            anchors.rightMargin: 16
            anchors.verticalCenter: parent.verticalCenter
            text: "Code changed since: restart the server"
            font.family: Theme.body; font.pixelSize: Theme.tiny; font.weight: Font.DemiBold
            color: Theme.warn
        }
    }

    // the opening picture, over the bar and the page, until it fades (components/Splash.qml)
    Splash { parent: Overlay.overlay; anchors.fill: parent; z: 1000 }

    ConfirmDialog { id: confirm; objectName: "confirm" }
    function ask(text, ok, danger, cancel, then) { confirm.ask(text, ok, danger, cancel, then) }
}
