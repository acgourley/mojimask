// Projects (web/app.html #s-start): start a project (choose, drop, or the sample clip), and the projects there are,
// with what each keeps on disk and its ⋯ menu. Everything shown is store.state.library; everything done is store.call.
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import QtQuick.Dialogs
import QtQuick.Shapes
import ".."
import "../components"

Rectangle {
    id: page
    objectName: "page-projects"
    color: Theme.bg
    readonly property var lib: store.state.library
    readonly property bool wide: width >= 980
    // File > Open (Main.qml's menu bar): the same chooser as the page's button
    function choose() { if (lib.ready && !lib.busy) chooser.open() }

    FileDialog {
        id: chooser
        objectName: "import-dialog"
        title: T.t("start.choose")
        nameFilters: [qsTr("Videos and photos (*.mp4 *.mov *.m4v *.jpg *.jpeg *.png *.heic)"), qsTr("All files (*)")]
        onAccepted: store.call("import", { path: selectedFile.toString() })
    }

    DropArea {
        anchors.fill: parent
        keys: ["text/uri-list"]
        onDropped: (drop) => { if (drop.hasUrls) { store.call("import", { path: drop.urls[0].toString() }); drop.accept() } }
    }

    ScrollView {
        id: scroll
        anchors.fill: parent
        contentWidth: availableWidth
        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

        ColumnLayout {
            id: col
            width: Math.min(1400, scroll.availableWidth - 64)
            x: (scroll.availableWidth - width) / 2
            spacing: 24
            Item { Layout.preferredHeight: 4 }

            // ---- the drop card
            Item {
                id: dropCard
                objectName: "drop"
                Layout.fillWidth: true
                implicitHeight: dropRow.implicitHeight + 72
                readonly property real gl: Math.abs(width * 0.342) + Math.abs(height * 0.94)
                Shape {
                    anchors.fill: parent
                    preferredRendererType: Shape.CurveRenderer
                    ShapePath {
                        // linear-gradient(200deg, ...): 200deg points down and a little left
                        strokeColor: dropZone.containsDrag ? Theme.accent : Theme.line3
                        strokeWidth: 1.5
                        fillGradient: LinearGradient {
                            x1: dropCard.width / 2 + 0.342 * dropCard.gl / 2
                            y1: dropCard.height / 2 - 0.94 * dropCard.gl / 2
                            x2: dropCard.width / 2 - 0.342 * dropCard.gl / 2
                            y2: dropCard.height / 2 + 0.94 * dropCard.gl / 2
                        GradientStop { position: 0; color: Theme.dropA }
                            GradientStop { position: 0.5; color: Theme.dropB }
                            GradientStop { position: 1; color: Theme.panel }
                        }
                        PathRectangle { x: 0.75; y: 0.75; width: dropCard.width - 1.5; height: dropCard.height - 1.5; radius: 20 }
                    }
                }
                DropArea {
                    id: dropZone
                    anchors.fill: parent
                    keys: ["text/uri-list"]
                    onDropped: (drop) => { if (drop.hasUrls) { store.call("import", { path: drop.urls[0].toString() }); drop.accept() } }
                }
                RowLayout {
                    id: dropRow
                    anchors.fill: parent
                    anchors.leftMargin: 40
                    anchors.rightMargin: 48
                    anchors.topMargin: 36
                    anchors.bottomMargin: 36
                    spacing: 32
                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 0
                        Text {
                            text: T.t("start.drop_title")
                            font.family: Theme.display; font.pixelSize: 34; font.weight: Font.ExtraBold
                            color: Theme.text
                            Layout.fillWidth: true
                            wrapMode: Text.Wrap
                            Layout.bottomMargin: 14
                        }
                        Text {
                            text: T.t("start.drop_sub")
                            font.family: Theme.body; font.pixelSize: 16
                            color: Theme.soft
                            Layout.fillWidth: true
                            wrapMode: Text.Wrap
                            Layout.bottomMargin: 24
                        }
                        Row {
                            spacing: 10
                            MmButton {
                                objectName: "pick"
                                variant: "primary"; big: true
                                iconName: "upload"
                                text: page.lib.busy || T.t("start.choose")
                                enabled: page.lib.ready && !page.lib.busy
                                onClicked: chooser.open()
                            }
                            MmButton {
                                objectName: "sample"
                                big: true
                                iconName: "play"
                                text: T.t("start.sample")
                                enabled: page.lib.ready && !page.lib.busy
                                onClicked: store.call("import_sample")
                            }
                        }
                    }
                    // three old paintings (public domain), their faces covered the way the app covers them
                    Row {
                        visible: page.wide
                        spacing: 16
                        Layout.alignment: Qt.AlignVCenter
                        FramedArt { file: "mona.jpg"; artW: 96; artH: 130; emoji: "😏"; ex: 0.242; ey: 0.172; esize: 38; erot: -6; anchors.verticalCenter: parent.verticalCenter }
                        FramedArt { file: "cavalier.jpg"; artW: 123; artH: 164; emoji: "🥸"; ex: 0.213; ey: 0.196; esize: 41; erot: -11; anchors.verticalCenter: parent.verticalCenter }
                        FramedArt { file: "pearl.jpg"; artW: 96; artH: 130; emoji: "🐱"; ex: 0.206; ey: 0.182; esize: 49; erot: 11; anchors.verticalCenter: parent.verticalCenter }
                    }
                }
            }

            // ---- the projects
            ColumnLayout {
                objectName: "recent"
                Layout.fillWidth: true
                spacing: 0
                visible: page.lib.projects.length > 0 || page.lib.error !== "" || !page.lib.ready || page.lib.message !== ""
                Text {
                    objectName: "library-error"
                    visible: page.lib.error !== ""
                    text: page.lib.error
                    color: Theme.warn
                    font.family: Theme.body; font.pixelSize: Theme.small
                    Layout.fillWidth: true
                    wrapMode: Text.Wrap
                }
                Row {
                    visible: !page.lib.ready && page.lib.error === ""
                    spacing: 10
                    Spinner { size: 13; anchors.verticalCenter: parent.verticalCenter }
                    Text { text: T.t("scan.getting_ready"); color: Theme.dim; font.family: Theme.body; font.pixelSize: Theme.small }
                }
                RowLayout {
                    visible: page.lib.projects.length > 0
                    Layout.fillWidth: true
                    spacing: 16
                    Row {
                        spacing: 10
                        Text {
                            text: T.t("projects.title")
                            font.family: Theme.display; font.pixelSize: Theme.h2; font.weight: Font.ExtraBold
                            color: Theme.text
                            anchors.verticalCenter: parent.verticalCenter
                        }
                        Rectangle {
                            anchors.verticalCenter: parent.verticalCenter
                            width: countText.implicitWidth + 18; height: countText.implicitHeight + 4; radius: 10
                            color: Theme.raised
                            Text { id: countText; anchors.centerIn: parent; objectName: "project-count"; text: page.lib.projects.length; font.family: Theme.body; font.pixelSize: 12; font.weight: Font.DemiBold; color: Theme.soft }
                        }
                    }
                    Item { Layout.fillWidth: true }
                    Text { text: page.lib.on_disk; font.family: Theme.body; font.pixelSize: Theme.small; color: Theme.dim }
                    AbstractButton {
                        id: folder
                        objectName: "showlib"
                        hoverEnabled: true
                        onClicked: store.call("reveal", { path: page.lib.projects_dir })
                        ToolTip.visible: hovered
                        ToolTip.delay: 600
                        ToolTip.text: T.t("projects.show_folder_tip")
                        Layout.maximumWidth: page.width * 0.46
                        contentItem: Row {
                            spacing: 7
                            Icon { name: "folder-open"; size: 14; color: folder.hovered ? Theme.text : Theme.dim; anchors.verticalCenter: parent.verticalCenter }
                            Text {
                                text: page.lib.projects_dir_short
                                font.family: Theme.body; font.pixelSize: 12
                                color: folder.hovered ? Theme.text : Theme.dim
                                elide: Text.ElideRight
                                width: Math.min(implicitWidth, folder.Layout.maximumWidth - 40)
                                anchors.verticalCenter: parent.verticalCenter
                            }
                        }
                        padding: 5; leftPadding: 10; rightPadding: 10
                        background: Rectangle { color: Theme.panel; border.color: Theme.line; radius: 8 }
                    }
                }
                Text {
                    visible: page.lib.projects.length > 0
                    text: T.t("projects.hint")
                    font.family: Theme.body; font.pixelSize: Theme.small
                    color: Theme.faint
                    Layout.fillWidth: true
                    Layout.topMargin: 8
                    wrapMode: Text.Wrap
                }
                Text {
                    objectName: "library-message"
                    visible: page.lib.message !== ""
                    text: page.lib.message
                    font.family: Theme.body; font.pixelSize: Theme.small
                    color: Theme.soft
                    Layout.fillWidth: true
                    Layout.topMargin: 10
                    wrapMode: Text.Wrap
                }
                MmButton {
                    objectName: "clearstale"
                    visible: page.lib.clear_stale !== ""
                    small: true
                    text: page.lib.clear_stale
                    Layout.topMargin: 10
                    onClicked: store.call("project_action", { act: "clear_stale" })
                }
                GridLayout {
                    objectName: "pgrid"
                    Layout.fillWidth: true
                    Layout.topMargin: 18
                    columns: page.width < 900 ? 1 : 2
                    columnSpacing: 12
                    rowSpacing: 12
                    Repeater {
                        model: page.lib.projects
                        delegate: ProjectCard {
                            required property var modelData
                            p: modelData
                            dev: page.lib.dev
                            Layout.fillWidth: true
                            Layout.preferredWidth: 1
                        }
                    }
                }
            }

            Row {
                Layout.topMargin: 4
                Layout.bottomMargin: 40
                spacing: 4
                Text { text: T.t("legal.line"); font.family: Theme.body; font.pixelSize: 12; color: Theme.faint }
                Text {
                    objectName: "licences"
                    text: T.t("legal.link"); font.family: Theme.body; font.pixelSize: 12; color: Theme.soft; font.underline: true
                    MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: Qt.openUrlExternally(noticesUrl) }
                }
            }
        }
    }
}
