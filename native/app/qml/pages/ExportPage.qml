// The Export step (web/app.html's export section): the preview (the video with its covers, drawn by the same code
// as the export, framed as the chosen format), the format, crop or fit, trim, the corner mark, higher-risk mode, the
// save dialog, progress, cancel and Show in Finder. State: store.state.export_page (export.rs publish_page).
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import QtQuick.Dialogs
import Mojimask.Native
import ".."
import "../components"

Rectangle {
    id: page
    objectName: "page-export"
    color: Theme.bg
    readonly property var xp: store.state.export_page || ({})
    readonly property var ed: store.state.editor
    readonly property var vid: ed.video
    readonly property bool running: xp.state === "running"

    FileDialog {
        id: saveDialog
        objectName: "export-dialog"
        title: page.xp.photo ? "Save the anonymized photo" : "Save the anonymized video"
        fileMode: FileDialog.SaveFile
        currentFolder: page.xp.save_dir ? "file://" + page.xp.save_dir : ""
        selectedFile: page.xp.save_dir ? "file://" + page.xp.save_dir + "/" + page.xp.file_name : ""
        defaultSuffix: page.xp.photo ? "jpg" : "mp4"
        nameFilters: page.xp.photo ? ["JPEG (*.jpg)", "PNG (*.png)"] : ["MPEG-4 (*.mp4)"]
        onAccepted: store.call("export_run", { out: selectedFile.toString() })
    }

    Flickable {
        anchors.fill: parent
        contentHeight: two.implicitHeight + 56
        clip: true
        boundsBehavior: Flickable.StopAtBounds
        RowLayout {
            id: two
            x: 32; y: 28
            width: Math.min(parent.width - 64, 1400)
            spacing: 32

            // ------------------------------------------------ the preview, framed as the export will be
            ColumnLayout {
                Layout.fillWidth: true
                Layout.alignment: Qt.AlignTop
                spacing: 12
                Rectangle {
                    id: frame
                    objectName: "export-preview"
                    Layout.fillWidth: true
                    // the output's shape
                    readonly property real ow: page.xp.out_w || page.vid.width || 16
                    readonly property real oh: page.xp.out_h || page.vid.height || 9
                    Layout.preferredHeight: Math.min(width * oh / ow, page.height * 0.62)
                    color: "black"
                    radius: 14
                    clip: true
                    Item {
                        id: out
                        anchors.centerIn: parent
                        readonly property real k: Math.min(parent.width / frame.ow, parent.height / frame.oh)
                        width: frame.ow * k
                        height: frame.oh * k
                        clip: true
                        // fill: the picture scaled to cover the output and cropped in the middle; fit: whole, over its own dim copy
                        readonly property real s: page.vid.width > 0 ? (page.xp.fit === "fit" && page.xp.preset !== "original"
                                                    ? Math.min(width / page.vid.width, height / page.vid.height)
                                                    : Math.max(width / page.vid.width, height / page.vid.height)) : 1
                        VideoItem {
                            id: video
                            objectName: "export-video"
                            visible: page.vid.ready && store.state.app.page === "export"
                            textureWidth: page.vid.width
                            textureHeight: page.vid.height
                            // shrunk through mipmaps, as the Editor's (EditorPage.qml)
                            layer.enabled: out.s < 0.75
                            layer.textureSize: Qt.size(page.vid.width, page.vid.height)
                            layer.mipmap: true
                            layer.smooth: true
                            width: page.vid.width * out.s
                            height: page.vid.height * out.s
                            anchors.centerIn: parent
                        }
                        // the corner mark (track.watermark): its height a share of the output's short side
                        Wordmark {
                            objectName: "export-mark"
                            visible: page.xp.mark === true
                            opacity: 0.85
                            size: Math.max(10, Math.round(Math.min(out.width, out.height) * 0.028))
                            anchors.right: parent.right; anchors.bottom: parent.bottom
                            anchors.rightMargin: Math.round(size * 0.8); anchors.bottomMargin: Math.round(size * 0.8)
                        }
                    }
                }
                RowLayout {
                    visible: !page.xp.photo
                    spacing: 12
                    AbstractButton {
                        objectName: "export-play"
                        implicitWidth: 40; implicitHeight: 40
                        onClicked: store.call("toggle_play")
                        background: Rectangle { radius: 20; color: Theme.accent }
                        contentItem: Item { Icon { anchors.centerIn: parent; name: page.ed.playing ? "pause" : "play"; color: Theme.ink; size: 18 } }
                    }
                    TimeSlider {
                        objectName: "export-position"
                        Layout.fillWidth: true
                        from: 0; to: Math.max(0, page.vid.frames - 1); stepSize: 1
                        value: page.ed.t
                        onMoved: store.call("seek", { t: Math.round(value) })
                    }
                    Text {
                        text: T.t("unit.s", { n: (page.ed.t / Math.max(1, page.vid.fps)).toFixed(1) })
                        font.family: Theme.body; font.pixelSize: Theme.small; font.features: { "tnum": 1 }; color: Theme.dim
                        Layout.minimumWidth: 48; horizontalAlignment: Text.AlignRight
                    }
                }
            }

            // ------------------------------------------------ the choices
            ColumnLayout {
                Layout.preferredWidth: 380
                Layout.maximumWidth: 380
                Layout.minimumWidth: 380
                Layout.alignment: Qt.AlignTop
                spacing: 16
                Text { text: T.t("export.title"); font.family: Theme.display; font.pixelSize: 30; font.weight: Font.ExtraBold; color: Theme.text }
                Text {
                    objectName: "export-styles"
                    Layout.topMargin: -8
                    textFormat: Text.StyledText
                    text: (page.xp.styles || "") + " · <a href='edit'>" + T.t("export.change_in_edit") + "</a>"
                    linkColor: Theme.accent
                    onLinkActivated: store.call("go", { page: "editor" })
                    font.family: Theme.body; font.pixelSize: Theme.small; color: Theme.soft
                    wrapMode: Text.WordWrap; Layout.fillWidth: true
                }
                Eyebrow { text: T.t("export.format") }
                GridLayout {
                    columns: 2
                    columnSpacing: 8; rowSpacing: 8
                    Layout.fillWidth: true
                    Repeater {
                        model: page.xp.presets || []
                        delegate: AbstractButton {
                            id: fb
                            required property var modelData
                            objectName: "format-" + modelData.id
                            Layout.fillWidth: true
                            Layout.preferredWidth: 1
                            hoverEnabled: true
                            onClicked: store.call("export_set", { preset: modelData.id })
                            padding: 12
                            contentItem: ColumnLayout {
                                spacing: 2
                                Text { text: fb.modelData.label; font.family: Theme.body; font.pixelSize: 14; font.weight: Font.DemiBold; color: fb.modelData.on ? Theme.accent : Theme.text }
                                Text { text: fb.modelData.detail; font.family: Theme.body; font.pixelSize: 12; color: Theme.dim; wrapMode: Text.WordWrap; Layout.fillWidth: true }
                            }
                            background: Rectangle { radius: 12; color: fb.modelData.on ? Theme.accentTint : Theme.panel; border.color: fb.modelData.on ? Theme.accent : fb.hovered ? Theme.line2 : Theme.line }
                        }
                    }
                }
                SegTabs {
                    objectName: "fit"
                    visible: page.xp.fit_shown === true
                    fill: true
                    Layout.fillWidth: true
                    keys: ["fill", "fit"]
                    labels: [T.t("export.fit_fill"), T.t("export.fit_fit")]
                    current: page.xp.fit || "fill"
                    onPicked: (k) => store.call("export_set", { fit: k })
                }
                RowLayout {
                    visible: !page.xp.photo
                    spacing: 14
                    Repeater {
                        model: [["start", "export.start", "0"], ["end", "export.end", T.t("export.end_placeholder")]]
                        delegate: RowLayout {
                            required property var modelData
                            spacing: 4
                            Text { text: T.t(modelData[1]); font.family: Theme.body; font.pixelSize: 13; color: Theme.dim }
                            TextField {
                                objectName: "export-" + modelData[0]
                                implicitWidth: 76
                                placeholderText: modelData[2]
                                text: page.xp[modelData[0]] || ""
                                validator: DoubleValidator { bottom: 0 }
                                color: Theme.text; placeholderTextColor: Theme.faint
                                font.family: Theme.body; font.pixelSize: 13
                                background: Rectangle { radius: 8; color: Theme.deep; border.color: parent.activeFocus ? Theme.accent : Theme.line }
                                onTextEdited: {
                                    const a = {}; a[modelData[0]] = text; store.call("export_set", a)
                                    if (modelData[0] === "start" && text !== "") store.call("seek", { t: Math.round(Number(text) * page.vid.fps) })
                                }
                            }
                            Text { text: T.t("export.seconds"); font.family: Theme.body; font.pixelSize: 13; color: Theme.dim }
                        }
                    }
                }
                Text { objectName: "export-summary"; text: page.xp.summary || ""; font.family: Theme.body; font.pixelSize: 12; font.features: { "tnum": 1 }; color: Theme.dim; wrapMode: Text.WordWrap; Layout.fillWidth: true }
                RowLayout {
                    spacing: 10
                    Toggle { objectName: "export-mark-switch"; checked: page.xp.mark === true; onClicked: store.call("export_set", { mark: checked }) }
                    Text { text: T.t("export.mark"); font.family: Theme.body; font.pixelSize: 14; color: Theme.soft }
                }
                Rectangle {
                    objectName: "export-strict"
                    Layout.fillWidth: true
                    implicitHeight: strictRow.implicitHeight + 28
                    radius: 12
                    color: Theme.panel
                    border.color: page.xp.strict ? Theme.accent : Theme.line
                    RowLayout {
                        id: strictRow
                        x: 16; y: 14; width: parent.width - 32
                        spacing: 12
                        Check { objectName: "export-strict-box"; checked: page.xp.strict === true; onClicked: store.call("export_set", { strict: checked }); Layout.alignment: Qt.AlignTop }
                        Text { textFormat: Text.StyledText; text: T.t("export.strict_html"); wrapMode: Text.WordWrap; Layout.fillWidth: true; font.family: Theme.body; font.pixelSize: 13; color: Theme.soft }
                    }
                    MouseArea { anchors.fill: parent; z: -1; onClicked: store.call("export_set", { strict: !page.xp.strict }) }
                }
                Text { visible: (page.xp.strict_note || "") !== ""; text: page.xp.strict_note || ""; color: Theme.warn; wrapMode: Text.WordWrap; Layout.fillWidth: true; font.family: Theme.body; font.pixelSize: 13 }
                Rectangle {
                    Layout.fillWidth: true
                    implicitHeight: note.implicitHeight + 20
                    color: "#14F5A524"
                    radius: 10
                    Rectangle { width: 3; height: parent.height; color: Theme.warn }
                    Text {
                        id: note
                        x: 14; y: 10; width: parent.width - 28
                        textFormat: Text.StyledText
                        text: page.xp.photo ? T.t("export.note_html").replace(/<span id="noteaudio">.*?<\/span>/, "") : T.t("export.note_html")
                        wrapMode: Text.WordWrap
                        font.family: Theme.body; font.pixelSize: 13; color: Theme.soft
                    }
                }
                RowLayout {
                    spacing: 12
                    MmButton {
                        objectName: "export-run"
                        variant: "primary"; big: true
                        iconName: "download"
                        text: T.t("export.button")
                        enabled: !page.running && page.xp.loaded === true
                        onClicked: saveDialog.open()
                    }
                    MmButton { objectName: "export-cancel"; visible: page.running; text: T.t("common.cancel"); onClicked: store.call("cancel_export") }
                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 4
                        Row {
                            spacing: 6
                            Text {
                                objectName: "export-status"
                                text: page.xp.note || page.xp.status || ""
                                color: page.xp.state === "failed" || page.xp.note ? Theme.bad : Theme.dim
                                font.family: Theme.body; font.pixelSize: 13
                            }
                            Text {
                                objectName: "export-reveal"
                                visible: page.xp.can_reveal === true
                                text: "· " + T.t("common.show_in_finder")
                                color: Theme.accent
                                font.family: Theme.body; font.pixelSize: 13
                                MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: store.call("export_reveal") }
                            }
                        }
                        ProgressLine { objectName: "export-progress"; visible: page.running; value: page.xp.progress || 0; Layout.fillWidth: true }
                    }
                }
            }
        }
    }
}
