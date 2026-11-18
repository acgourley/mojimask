// The result after a project's first scan (web/app.html's s-result, startResult): the clip's busiest frame with a ring
// on each face, the steps as they happen; then the same frame fades to its covered version (drawn by the export's own
// code: the VideoItem), the video plays on from there, and "Nice, it worked!" appears with the styles, the two ways on
// and confetti. What it shows is store.state.result (library.rs result_view); the timing is this page's.
import QtQuick
import QtQuick.Window
import QtQuick.Controls.Basic
import QtQuick.Layouts
import Mojimask.Native
import ".."
import "../components"

Rectangle {
    id: page
    objectName: "page-result"
    color: Theme.bg
    readonly property var r: store.state.result || ({})
    readonly property var ed: store.state.editor
    readonly property bool shown: store.state.app.page === "result"
    // 0: covering (people), 1: covering (covers), 2: done (fading), 3: the result
    property int phase: 0
    readonly property bool reduced: false

    function start() {
        phase = 0
        store.call("pause")
        store.call("seek", { t: r.t || 0 })
        t1.restart()
    }
    onShownChanged: if (shown) Qt.callLater(start)
    Component.onCompleted: if (shown) Qt.callLater(start)

    Timer { id: t1; interval: 1300; onTriggered: { page.phase = 1; t2.restart() } }
    // the covers are in once the frame is drawn from the records (and at least 2 s since the start, as app.html paces it)
    Timer {
        id: t2; interval: 2000
        onTriggered: wait.restart()
    }
    Timer {
        id: wait; interval: 100; repeat: true
        onTriggered: if (page.ed.records > 0 && page.ed.shown === (page.r.t || 0)) { stop(); page.phase = 2; t3.restart() }
    }
    Timer { id: t3; interval: 1100; onTriggered: { page.phase = 3; store.call("play"); confetti.go() } }

    Flickable {
        anchors.fill: parent
        contentHeight: col.implicitHeight + 48
        clip: true
        boundsBehavior: Flickable.StopAtBounds
        ColumnLayout {
            id: col
            x: (parent.width - width) / 2; y: 24
            width: Math.min(800, parent.width - 48)
            spacing: 0

            // the stage: the frame at its own shape, as big as the player will be
            Rectangle {
                id: stage
                objectName: "result-stage"
                Layout.fillWidth: true
                Layout.preferredHeight: Math.min(page.height * 0.58, width * (page.r.h || 9) / (page.r.w || 16))
                radius: 14
                color: "black"
                clip: true
                Item {
                    id: fr
                    anchors.centerIn: parent
                    readonly property real k: Math.min(parent.width / (page.r.w || 16), parent.height / (page.r.h || 9))
                    width: (page.r.w || 16) * k
                    height: (page.r.h || 9) * k
                    VideoItem {
                        objectName: "result-video"
                        anchors.fill: parent
                        visible: page.shown && page.ed.video.ready
                        textureWidth: page.ed.video.width
                        textureHeight: page.ed.video.height
                    }
                    // the raw frame (app.html: saturate .75, brightness .8), fading out as the covers come in
                    Image {
                        objectName: "result-raw"
                        anchors.fill: parent
                        source: page.r.raw || ""
                        fillMode: Image.PreserveAspectFit
                        sourceSize: Qt.size(width * Screen.devicePixelRatio, height * Screen.devicePixelRatio)
                        smooth: true
                        opacity: page.phase >= 2 ? 0 : 1
                        Behavior on opacity { NumberAnimation { duration: 900; easing.type: Easing.InOutQuad } }
                        Rectangle { anchors.fill: parent; color: "black"; opacity: 0.2 }
                    }
                    Repeater {
                        model: page.r.rings || []
                        delegate: Rectangle {
                            id: ring
                            required property var modelData
                            x: modelData.x * fr.k; y: modelData.y * fr.k
                            width: modelData.d * fr.k; height: width
                            radius: width / 2
                            color: "transparent"
                            border.width: 2
                            border.color: Theme.accent
                            opacity: page.phase >= 2 ? 0 : 1
                            Behavior on opacity { NumberAnimation { duration: 500 } }
                            Rectangle { anchors.fill: parent; anchors.margins: -4; radius: width / 2; color: "transparent"; border.width: 4; border.color: "#40FFD23F" }
                            SequentialAnimation on scale {
                                loops: Animation.Infinite
                                running: page.phase < 2 && page.shown
                                PauseAnimation { duration: Math.random() * 1600 }
                                NumberAnimation { from: 0.92; to: 1.08; duration: 800; easing.type: Easing.InOutSine }
                                NumberAnimation { from: 1.08; to: 0.92; duration: 800; easing.type: Easing.InOutSine }
                            }
                        }
                    }
                }
            }

            // the covering step: the head and the steps
            ColumnLayout {
                visible: page.phase < 3
                Layout.alignment: Qt.AlignHCenter
                spacing: 6
                Text {
                    objectName: "result-covering"
                    Layout.alignment: Qt.AlignHCenter
                    Layout.topMargin: 20
                    text: page.phase >= 2 ? page.r.done || "" : page.r.covering || ""
                    font.family: Theme.display; font.pixelSize: 26; font.weight: Font.ExtraBold; color: Theme.text
                }
                Repeater {
                    model: page.r.steps || []
                    delegate: RowLayout {
                        required property string modelData
                        required property int index
                        // the step on show: 1 (people) while covering starts, 2 (covers) after; done marks them all
                        readonly property int at: page.phase === 0 ? 1 : 2
                        readonly property string st: index < at || (index === at && page.phase >= 2) ? "ok" : index === at ? "on" : ""
                        spacing: 6
                        Item {
                            implicitWidth: 22; implicitHeight: 18
                            Text { anchors.centerIn: parent; visible: parent.parent.st !== "on"; text: parent.parent.st === "ok" ? "✓" : "·"; color: parent.parent.st === "ok" ? Theme.good : Theme.dim; font.pixelSize: 14 }
                            Spinner { anchors.centerIn: parent; visible: parent.parent.st === "on"; size: 14 }
                        }
                        Text { text: modelData; font.family: Theme.body; font.pixelSize: 14; color: parent.st === "" ? Theme.dim : Theme.text }
                    }
                }
            }

            // the result: the player's controls, the head, the styles, the two ways on
            RowLayout {
                visible: page.phase >= 3 && !page.r.photo
                Layout.topMargin: 12
                spacing: 12
                AbstractButton {
                    objectName: "result-play"
                    implicitWidth: 40; implicitHeight: 40
                    onClicked: store.call("toggle_play")
                    background: Rectangle { radius: 20; color: Theme.accent }
                    contentItem: Item { Icon { anchors.centerIn: parent; name: page.ed.playing ? "pause" : "play"; color: Theme.ink; size: 18 } }
                }
                TimeSlider {
                    Layout.fillWidth: true
                    from: 0; to: Math.max(0, page.ed.video.frames - 1); stepSize: 1
                    value: page.ed.t
                    onMoved: store.call("seek", { t: Math.round(value) })
                }
                Text {
                    text: T.t("unit.s", { n: (page.ed.t / Math.max(1, page.ed.video.fps)).toFixed(1) })
                    font.family: Theme.body; font.pixelSize: Theme.small; font.features: { "tnum": 1 }; color: Theme.dim
                    Layout.minimumWidth: 48; horizontalAlignment: Text.AlignRight
                }
            }
            ColumnLayout {
                id: hero
                objectName: "result-hero"
                visible: page.phase >= 3
                Layout.fillWidth: true
                Layout.topMargin: 22
                spacing: 6
                opacity: visible ? 1 : 0
                Behavior on opacity { NumberAnimation { duration: 500 } }
                Text { Layout.alignment: Qt.AlignHCenter; text: page.r.head || ""; font.family: Theme.display; font.pixelSize: 34; font.weight: Font.ExtraBold; color: Theme.text }
                Text { Layout.alignment: Qt.AlignHCenter; Layout.bottomMargin: 10; text: page.r.sub || ""; font.family: Theme.body; font.pixelSize: 15; color: Theme.soft }
                GridLayout {
                    objectName: "result-styles"
                    Layout.fillWidth: true
                    columns: Math.max(1, Math.min((page.r.styles || []).length, Math.floor(width / 150)))
                    columnSpacing: 10; rowSpacing: 10
                    Repeater {
                        model: page.r.styles || []
                        delegate: AbstractButton {
                            id: sb
                            required property var modelData
                            objectName: "result-style-" + modelData.k
                            Layout.fillWidth: true
                            Layout.preferredWidth: 1
                            hoverEnabled: true
                            padding: 12
                            onClicked: if (!modelData.on) store.call("result_style", { style: modelData.k })
                            contentItem: GridLayout {
                                columns: 2
                                columnSpacing: 10; rowSpacing: 2
                                Text { text: sb.modelData.icon; font.pixelSize: 20; color: Theme.text }
                                Text { Layout.fillWidth: true; text: sb.modelData.name; font.family: Theme.body; font.pixelSize: 14; font.weight: Font.DemiBold; color: sb.modelData.on ? Theme.accent : Theme.text }
                                Text { Layout.columnSpan: 2; Layout.fillWidth: true; text: sb.modelData.hint; wrapMode: Text.WordWrap; font.family: Theme.body; font.pixelSize: 12; color: Theme.dim }
                            }
                            background: Rectangle { radius: 12; color: sb.modelData.on ? Theme.accentTint : Theme.panel; border.color: sb.modelData.on ? Theme.accent : sb.hovered ? Theme.line2 : Theme.line }
                        }
                    }
                }
                RowLayout {
                    Layout.fillWidth: true
                    Layout.topMargin: 8
                    spacing: 12
                    Repeater {
                        model: [["export", true], ["decisions", false]]
                        delegate: AbstractButton {
                            id: nb
                            required property var modelData
                            objectName: "result-" + (modelData[0] === "export" ? "export" : "refine")
                            Layout.fillWidth: true
                            Layout.preferredWidth: 1
                            hoverEnabled: true
                            padding: 14; leftPadding: 18
                            onClicked: store.call("result_done", { to: modelData[0] })
                            contentItem: ColumnLayout {
                                spacing: 3
                                Text { text: (nb.modelData[1] ? page.r.export : page.r.refine) || ""; font.family: Theme.body; font.pixelSize: 15; font.weight: Font.Bold; color: nb.modelData[1] ? Theme.ink : Theme.text }
                                Text { Layout.fillWidth: true; wrapMode: Text.WordWrap; text: (nb.modelData[1] ? page.r.export_sub : page.r.refine_sub) || ""; font.family: Theme.body; font.pixelSize: 13; color: nb.modelData[1] ? "#B31A1405" : Theme.dim }
                            }
                            background: Rectangle { radius: 12; color: nb.modelData[1] ? Theme.accent : (nb.hovered ? Theme.raised : Theme.panel); border.color: nb.modelData[1] ? Theme.accent : Theme.line2 }
                        }
                    }
                }
            }
        }
    }

    // confetti (app.html confetti()): 160 bits from the middle, gravity, fading after 1.6 s, gone at 2.8 s
    Canvas {
        id: confetti
        objectName: "confetti"
        anchors.fill: parent
        z: 10
        visible: false
        property var bits: []
        property real t0: 0
        function go() {
            const colours = ["#FFD23F", "#3ddc84", "#ffb020", "#ff5c5c", "#c77dff", "#ffffff"]
            const b = []
            for (let i = 0; i < 160; i++) b.push({ x: width / 2 + (Math.random() - 0.5) * 160, y: height * 0.35, vx: (Math.random() - 0.5) * 14,
                vy: -6 - Math.random() * 10, r: Math.random() * 6.3, vr: (Math.random() - 0.5) * 0.4, w: 6 + Math.random() * 6,
                h: 3 + Math.random() * 5, c: colours[Math.floor(Math.random() * colours.length)] })
            bits = b; t0 = Date.now(); visible = true; tick.start()
        }
        Timer {
            id: tick; interval: 16; repeat: true
            onTriggered: { if ((Date.now() - confetti.t0) / 1000 > 2.8) { stop(); confetti.visible = false } else confetti.requestPaint() }
        }
        onPaint: {
            const g = getContext("2d"), age = (Date.now() - t0) / 1000
            g.clearRect(0, 0, width, height)
            for (const b of bits) {
                b.vy += 0.32; b.vx *= 0.99; b.x += b.vx; b.y += b.vy; b.r += b.vr
                g.save(); g.globalAlpha = Math.max(0, 1 - Math.max(0, age - 1.6) / 1.2); g.translate(b.x, b.y); g.rotate(b.r)
                g.fillStyle = b.c; g.fillRect(-b.w / 2, -b.h / 2, b.w, b.h); g.restore()
            }
        }
    }
}
