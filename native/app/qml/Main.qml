// The window: the bar (web/app.html's titlebar: the wordmark and where you are on the left, the project's steps in
// the middle; it is also the Mac title bar) and the page shown. State and actions are the store's (store.state,
// store.call); this file only lays them out.
import QtQuick
import QtQuick.Controls.Basic
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

    // the Mac menu bar's Undo / Redo (app.html menuCommand): the project's history on its server
    Shortcut { sequences: [StandardKey.Undo]; enabled: win.project.open && win.page !== "editor"; onActivated: store.call("undo") }
    Shortcut { sequences: [StandardKey.Redo]; enabled: win.project.open && win.page !== "editor"; onActivated: store.call("redo") }

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
        ProjectsPage {}
        ScanPage {}
        DecisionsPage {}
        PeoplePage {}
        EditorPage {}
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
