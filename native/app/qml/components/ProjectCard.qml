// A project in the list (app.html .proj): its picture, name, what it is and what it keeps on disk; Open (or Resume,
// or Locate video…) and the ⋯ menu of the things done rarely.
import QtQuick
import QtQuick.Window
import QtQuick.Controls.Basic
import QtQuick.Layouts
import QtQuick.Dialogs
import QtQuick.Effects
import ".."

Rectangle {
    id: card
    property var p
    property bool dev: false
    objectName: "project-" + p.id
    implicitHeight: row.implicitHeight + 24
    radius: 14
    color: Theme.panel
    border.color: hover.hovered ? Theme.line3 : Theme.line
    HoverHandler { id: hover }

    FileDialog {
        id: locate
        title: T.t("projects.locate.ask", { name: card.p.name })
        onAccepted: store.call("project_action", { id: card.p.id, act: "relink", path: selectedFile.toString() })
    }

    RowLayout {
        id: row
        anchors.fill: parent
        anchors.margins: 12
        spacing: 14
        // its picture: a frame with its faces covered, saved when its first scan finished
        Picture {
            objectName: "thumb-" + card.p.id
            mmPublic: card.p.public
            Layout.preferredWidth: 96
            Layout.preferredHeight: 68
            Rectangle {
                id: thumbBox
                anchors.fill: parent
                radius: 8
                color: Theme.chip
                border.color: card.p.thumb ? "transparent" : Theme.line
                Icon { visible: !card.p.thumb; anchors.centerIn: parent; name: "scan-face"; size: 22; color: Theme.faint }
            }
            Image {
                id: thumb
                visible: false
                anchors.fill: parent
                source: card.p.thumb
                fillMode: Image.PreserveAspectCrop
                sourceSize: Qt.size(width * Screen.devicePixelRatio, height * Screen.devicePixelRatio)
                asynchronous: true
                smooth: true
            }
            Rectangle { id: thumbMask; anchors.fill: parent; radius: 8; visible: false; layer.enabled: true }
            MultiEffect {
                visible: card.p.thumb !== ""
                anchors.fill: parent
                source: thumb
                maskEnabled: true
                maskSource: thumbMask
                maskThresholdMin: 0.5
                maskSpreadAtMin: 1.0
            }
        }
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 4
            Text {
                objectName: "name"
                text: card.p.name
                font.family: Theme.body; font.pixelSize: 15; font.weight: Font.DemiBold
                color: Theme.text
                elide: Text.ElideRight
                Layout.fillWidth: true
            }
            Text { text: card.p.meta; font.family: Theme.body; font.pixelSize: 12; color: Theme.dim; Layout.fillWidth: true; elide: Text.ElideRight }
            Text {
                text: card.p.sizes
                font.family: Theme.body; font.pixelSize: 12
                color: card.p.video_found ? Theme.faint : Theme.warn
                elide: Text.ElideRight
                Layout.fillWidth: true
                HoverHandler { id: sizesHover }
                ToolTip.visible: sizesHover.hovered
                ToolTip.delay: 600
                ToolTip.text: card.p.sizes_tip
            }
        }
        MmButton {
            objectName: "open-" + card.p.id
            variant: "raised"
            leftPadding: 16; rightPadding: 16
            text: card.p.open_label
            onClicked: card.p.video_found ? store.call("open_project", { id: card.p.id }) : locate.open()
        }
        MmButton {
            id: more
            objectName: "more-" + card.p.id
            variant: "icon"
            iconName: "ellipsis"
            iconSize: 18
            onClicked: menu.opened ? menu.close() : menu.open()
            Accessible.name: T.t("projects.more_for", { name: card.p.name })
        }
    }

    Popup {
        id: menu
        objectName: "menu-" + card.p.id
        x: card.width - width - 12
        y: 60
        width: 236
        padding: 6
        background: Rectangle {
            color: Theme.menu; border.color: Theme.line3; radius: 12
            RectangularShadow { z: -1; anchors.fill: parent; offset.y: 12; blur: 32; color: "#B3000000"; radius: 12 }
        }
        contentItem: Column {
            spacing: 2
            MenuRow { objectName: "menu-clear"; ico: "eraser"; label: T.t("projects.menu.clear"); tag: card.p.cache_text; enabled: card.p.cache > 0
                      tip: T.t("projects.menu.clear_tip"); onPicked: store.call("project_action", { id: card.p.id, act: "clear" }) }
            MenuRow { objectName: "menu-reveal"; ico: "folder-open"; label: T.t("common.show_in_finder"); onPicked: store.call("reveal", { path: card.p.path }) }
            MenuRow { objectName: "menu-clone"; visible: card.p.scanned && card.dev; ico: "copy"; label: "Clone fresh"; tag: "DEV"
                      tip: "Developer mode: a copy with the same scan but none of your answers, opening on the result screen like a new project"
                      onPicked: store.call("project_action", { id: card.p.id, act: "clone" }) }
            Rectangle { width: parent.width; height: 1; color: Theme.line3 }
            MenuRow {
                objectName: "menu-delete"; ico: "trash-2"; label: T.t("projects.menu.delete"); danger: true; tip: T.t("projects.menu.delete_tip")
                onPicked: win.ask(T.t("projects.delete.ask", { name: card.p.name, video: card.p.copied ? T.t("projects.delete.copy", { size: card.p.copy_text })
                                                                                                    : T.t("projects.delete.original", { path: card.p.video }) }),
                                  T.t("projects.delete.ok"), true, null, () => store.call("project_action", { id: card.p.id, act: "delete" }))
            }
        }
    }

    component MenuRow: AbstractButton {
        id: mr
        property string ico
        property string label
        property string tag
        property string tip
        property bool danger: false
        signal picked()
        width: 224
        height: visible ? 34 : 0
        hoverEnabled: true
        opacity: enabled ? 1 : 0.45
        onClicked: { menu.close(); picked() }
        ToolTip.visible: hovered && tip !== ""
        ToolTip.delay: 800
        ToolTip.text: tip
        background: Rectangle { radius: 8; color: mr.hovered && mr.enabled ? Theme.raised2 : "transparent" }
        contentItem: RowLayout {
            spacing: 10
            Icon { Layout.leftMargin: 10; name: mr.ico; size: 15; color: mr.danger ? Theme.bad : Theme.dim }
            Text { text: mr.label; font.family: Theme.body; font.pixelSize: Theme.small; color: mr.danger ? Theme.bad : Theme.text; Layout.fillWidth: true }
            Text { visible: mr.tag !== ""; text: mr.tag; Layout.rightMargin: 10; font.family: Theme.body; font.pixelSize: 11; font.weight: Font.DemiBold; font.letterSpacing: 0.5; color: Theme.faint }
        }
    }
}
