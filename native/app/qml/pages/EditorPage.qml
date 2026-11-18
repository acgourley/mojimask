// The Editor (web/review.html and its plug-ins light_ui.js, emoji_set_ui.js, pose_ui.js, depth_ui.js): the video with
// its covers drawn by Rust (mm-render's Drawer, the export's own drawing) into the VideoItem; the boxes, labels and
// selection drawn here over it from store.state.editor.view; the panel (cover style, selection, emoji preset,
// lighting). Everything it shows is the store's state; everything it does is store.call (editor_* commands).
// Without a project (--video/--data: a PoC dump) it's the old dump editor: drag a cover's circle, export.
import QtQuick
import QtQuick.Window
import QtQuick.Controls.Basic
import QtQuick.Layouts
import QtQuick.Dialogs
import Mojimask.Native
import ".."
import "../components"

Rectangle {
    id: page
    objectName: "page-editor"
    color: Theme.bg
    readonly property var ed: store.state.editor
    readonly property var vid: ed.video
    readonly property var ex: store.state.export
    readonly property bool proj: ed.project === true
    readonly property var vw: ed.view || ({})
    readonly property bool active: store.state.app.page === "editor"

    function kindColour(k) { return k === "coast" ? Theme.coast : k === "manual" ? Theme.manual : Theme.obs }
    function key(k, shift) { store.call("editor_key", { key: k, shift: !!shift }) }

    FileDialog {
        id: openDialog
        objectName: "open-dialog"
        title: qsTr("Open Video")
        nameFilters: [qsTr("Videos (*.mp4 *.mov *.m4v)")]
        onAccepted: store.call("open", { video: selectedFile.toString(), data: page.ed.data })
    }
    FileDialog {
        id: saveDialog
        objectName: "export-dialog"
        title: qsTr("Export Video")
        fileMode: FileDialog.SaveFile
        defaultSuffix: "mp4"
        nameFilters: [qsTr("MPEG-4 (*.mp4)")]
        onAccepted: store.call("export", { out: selectedFile.toString() })
    }
    FileDialog {
        id: photoDialog
        objectName: "portrait-dialog"
        title: T.t("edit.portrait.choose")
        nameFilters: ["PNG (*.png)", "Images (*.png *.jpg *.jpeg *.heic)"]
        onAccepted: store.call("editor_portrait", { path: selectedFile.toString() })
    }

    // keys: review.html's, sent to the store (editor_key); typing in a field keeps its own
    readonly property bool keysOn: active && !(addField.activeFocus)
    Shortcut { enabled: page.keysOn; sequence: "Space"; onActivated: page.proj ? page.key("Space") : store.call("toggle_play") }
    Shortcut { enabled: page.keysOn; sequence: "Right"; onActivated: page.proj ? page.key("Right") : store.call("step", { d: 1 }) }
    Shortcut { enabled: page.keysOn; sequence: "Left"; onActivated: page.proj ? page.key("Left") : store.call("step", { d: -1 }) }
    Shortcut { enabled: page.keysOn && page.proj; sequence: "Shift+Right"; onActivated: page.key("Right", true) }
    Shortcut { enabled: page.keysOn && page.proj; sequence: "Shift+Left"; onActivated: page.key("Left", true) }
    Repeater {
        model: ["V", "D", "B", "Escape", "Delete", "Backspace", "S", "R", "[", "]", "K", "E", "U", "?", "Shift+?"]
        delegate: Item {
            required property string modelData
            Shortcut { enabled: page.keysOn && page.proj; sequence: modelData; onActivated: page.key(modelData === "Shift+?" ? "?" : modelData) }
        }
    }
    Shortcut { enabled: page.active && page.proj; sequences: [StandardKey.Undo]; onActivated: store.call("undo") }
    Shortcut { enabled: page.active && page.proj; sequences: [StandardKey.Redo]; onActivated: store.call("redo") }
    Shortcut { enabled: page.active; sequences: [StandardKey.Open]; onActivated: if (!page.proj) openDialog.open() }
    Shortcut { enabled: page.active; sequence: "Ctrl+E"; onActivated: if (!page.proj) saveDialog.open() }

    RowLayout {
        anchors.fill: parent
        spacing: 0

        // ---------------------------------------------------------------- the picture's column
        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.margins: 20
            spacing: 14

            // the PoC dump editor's bar: what's open, the export
            RowLayout {
                visible: !page.proj
                Layout.fillWidth: true
                spacing: 10
                MmButton { objectName: "open-button"; text: qsTr("Open…"); onClicked: openDialog.open() }
                ColumnLayout {
                    spacing: 0
                    Layout.fillWidth: true
                    Text {
                        objectName: "video-name"
                        text: page.vid.name || qsTr("No video open")
                        font.family: Theme.display; font.pixelSize: 17; font.weight: Font.ExtraBold
                        color: Theme.text
                        elide: Text.ElideMiddle
                        Layout.fillWidth: true
                    }
                    Text {
                        objectName: "status"
                        text: page.ed.status
                        font.family: Theme.body; font.pixelSize: Theme.tiny
                        color: Theme.dim
                        elide: Text.ElideRight
                        Layout.fillWidth: true
                    }
                }
                Chip { objectName: "edits-chip"; visible: page.ed.edits > 0; text: page.ed.edits === 1 ? qsTr("1 edit") : qsTr("%1 edits").arg(page.ed.edits) }
                Text {
                    objectName: "export-message"
                    visible: page.ex.state !== "idle"
                    text: page.ex.state === "running" ? qsTr("Exporting… %1 of %2").arg(page.ex.done).arg(page.ex.total) : page.ex.message
                    font.family: Theme.body; font.pixelSize: Theme.tiny
                    color: page.ex.state === "failed" ? Theme.bad : page.ex.state === "done" ? Theme.good : Theme.soft
                    elide: Text.ElideMiddle
                    Layout.maximumWidth: 360
                }
                MmButton {
                    objectName: "export-button"
                    variant: "primary"
                    text: qsTr("Export…")
                    enabled: page.vid.ready && page.ex.state !== "running"
                    onClicked: saveDialog.open()
                }
            }

            // the project editor's toolbar: covers / boxes, pose, depth; undo, redo
            RowLayout {
                visible: page.proj
                Layout.fillWidth: true
                spacing: 10
                SegTabs {
                    objectName: "view"
                    accentOn: true
                    keys: ["covers", "boxes"]
                    labels: [T.t("edit.view_covers"), T.t("edit.view_boxes")]
                    tips: ["V", "V"]
                    current: page.vw.view || "covers"
                    onPicked: (k) => { if (k !== page.vw.view) store.call("editor_toggle", { what: "view" }) }
                }
                MmButton {
                    objectName: "pose-toggle"
                    iconName: "person-standing"; iconSize: 14
                    text: T.t("edit.pose") + "  " + (page.vw.pose_view ? T.t("edit.toggle_on") : T.t("edit.toggle_off"))
                    tip: T.t("edit.pose_tip")
                    onClicked: store.call("editor_toggle", { what: "pose" })
                }
                MmButton {
                    objectName: "depth-toggle"
                    visible: page.vw.depth_on === true
                    iconName: "layers"; iconSize: 14
                    text: T.t("edit.depth") + "  " + (page.vw.depth_view ? T.t("edit.toggle_on") : T.t("edit.toggle_off"))
                    tip: T.t("edit.depth_tip")
                    onClicked: store.call("editor_toggle", { what: "depth" })
                }
                Item { Layout.fillWidth: true }
                Text {
                    objectName: "editor-error"
                    visible: (page.vw.error || "") !== ""
                    text: page.vw.error || ""
                    color: Theme.bad; font.family: Theme.body; font.pixelSize: Theme.tiny
                    elide: Text.ElideRight
                    Layout.maximumWidth: 360
                }
                Text { objectName: "label-status"; visible: page.vw.dev === true && (page.vw.labels || "") !== ""; text: page.vw.labels || ""; color: Theme.dim; font.family: Theme.body; font.pixelSize: 12 }
                MmButton { objectName: "export-labels"; visible: page.vw.dev === true; text: "Export labels"; tip: "Write clean face / not-face labels for training"; onClicked: store.call("editor_export_labels") }
                MmButton { objectName: "undo"; iconName: "undo-2"; iconSize: 14; text: T.t("edit.undo"); tip: "⌘Z"; enabled: store.state.project.can_undo; onClicked: store.call("undo") }
                MmButton { objectName: "redo"; iconName: "redo-2"; iconSize: 14; text: T.t("edit.redo"); tip: "⇧⌘Z"; enabled: store.state.project.can_redo; onClicked: store.call("redo") }
            }

            // the stage
            Item {
                id: stage
                objectName: "stage"
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true

                Text {
                    anchors.centerIn: parent
                    visible: !page.vid.path
                    text: qsTr("Drop a video here, or Open…")
                    font.family: Theme.body; font.pixelSize: Theme.textSize
                    color: Theme.faint
                }

                Rectangle {
                    anchors.fill: video
                    color: "black"
                    radius: 12
                    visible: page.vid.ready
                }
                VideoItem {
                    id: video
                    objectName: "video"
                    visible: page.vid.ready
                    clip: true
                    textureWidth: page.vid.width
                    textureHeight: page.vid.height
                    // drawn at the video's own size; shown through a mipmapped layer that size, so a 4K frame shrinks
                    // into the window smoothly (QQuickRhiItem has no mipmaps of its own; without them detail shimmers)
                    layer.enabled: k < 0.75
                    layer.textureSize: Qt.size(page.vid.width, page.vid.height)
                    layer.mipmap: true
                    layer.smooth: true
                    // aspect fit, at most 2.5x (review.html layout)
                    readonly property real k: page.vid.width > 0 ? Math.min((stage.width - 16) / page.vid.width, (stage.height - 16) / page.vid.height, page.proj ? 2.5 : 100) : 1
                    width: page.vid.width * k
                    height: page.vid.height * k
                    anchors.centerIn: parent

                    // the depth view: the depth map washed over the picture
                    Image {
                        objectName: "depth-pic"
                        anchors.fill: parent
                        visible: page.proj && page.vw.depth_view === true
                        source: visible ? page.vw.depth_pic : ""
                        sourceSize: Qt.size(width * Screen.devicePixelRatio, height * Screen.devicePixelRatio)
                        smooth: true
                        opacity: 0.55
                        cache: false
                        // a new frame's map arrives over HTTP: keep the last one until it has (cleared first, the
                        // map was missing ~16% of the time while playing a 4K clip: it flickered)
                        retainWhileLoading: true
                    }

                    // the project editor's boxes (video px * k), small first
                    Repeater {
                        model: page.proj ? (page.vw.boxes || []).length : 0
                        delegate: Item {
                            id: bx
                            required property int index
                            readonly property var b: page.vw.boxes[index] || ({ x: 0, y: 0, s: 0 })
                            objectName: "box-" + b.piece
                            x: (b.x - b.s / 2) * video.k
                            y: (b.y - b.s / 2) * video.k
                            width: b.s * video.k
                            height: width
                            opacity: b.hidden ? b.opacity : 1
                            readonly property color col: b.hidden ? (b.sel ? "#ffffff" : b.colour)
                                                       : b.shown ? (b.sel ? "#ffffff" : "#c9c9c9")
                                                       : b.sel ? "#ffffff" : b.same ? "#bbbbbb" : page.kindColour(b.kind)
                            DashBox {
                                anchors.fill: parent
                                stroke: bx.col
                                lineWidth: bx.b.sel ? (bx.b.hidden ? 2.5 : 3) : 1.5
                                dash: bx.b.hidden ? [3, 4] : bx.b.shown ? [2, 3] : (bx.b.kind === "coast" || bx.b.same) ? [4, 3] : []
                            }
                            Rectangle {
                                visible: (bx.b.label || "") !== ""
                                y: -14
                                width: lbl.implicitWidth + 6
                                height: 14
                                color: "#99000000"
                                Text { id: lbl; x: 3; anchors.verticalCenter: parent.verticalCenter; text: bx.b.label || ""; font.pixelSize: 11; color: bx.col }
                            }
                        }
                    }

                    // the depth view's labels: what partial occlusion decided for each cover
                    Repeater {
                        model: page.proj && page.vw.depth_view && page.vw.depth ? (page.vw.depth.covers || []).length : 0
                        delegate: Item {
                            required property int index
                            readonly property var c: page.vw.depth.covers[index]
                            readonly property real h: c.box[2] * page.vw.scale / 2
                            readonly property string what: c.face === undefined ? "drawn, never cut" : c.self ? "self" : c.cut ? "cut" : "not cut"
                            readonly property color col: what === "cut" ? "#7CFC00" : what === "self" ? "#ff5aa0" : "#ffffff"
                            x: (c.box[0] - h) * video.k
                            y: (c.box[1] - h) * video.k
                            width: 2 * h * video.k
                            height: width
                            DashBox { anchors.fill: parent; stroke: parent.col }
                            Rectangle {
                                y: -15
                                width: dl.implicitWidth + 4; height: 14
                                color: "#A6000000"
                                Text {
                                    id: dl; x: 2; anchors.verticalCenter: parent.verticalCenter; font.pixelSize: 11; color: parent.parent.col
                                    text: parent.parent.c.face === undefined ? parent.parent.what
                                        : parent.parent.what + " · face " + (parent.parent.c.face >= 0 ? "+" : "") + parent.parent.c.face.toFixed(2)
                                          + " · in front " + Math.round(parent.parent.c.in_front * 100) + "% · margin " + parent.parent.c.margin + "/" + parent.parent.c.rim
                                }
                            }
                        }
                    }

                    // the pose overlay: every skeleton the scan found here
                    Canvas {
                        id: poseCanvas
                        objectName: "pose-overlay"
                        anchors.fill: parent
                        visible: page.proj && page.vw.pose_view === true
                        readonly property var ps: page.vw.pose || []
                        onPsChanged: requestPaint()
                        onVisibleChanged: requestPaint()
                        onWidthChanged: requestPaint()
                        onPaint: {
                            const g = getContext("2d"), k = video.k
                            g.clearRect(0, 0, width, height)
                            if (!visible) return
                            const E = [[5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [0, 1], [0, 2], [1, 3], [2, 4]]
                            g.strokeStyle = "#bbbbbb"; g.fillStyle = "#bbbbbb"; g.lineWidth = 2
                            for (const p of ps) {
                                const q = p.kps
                                for (const [i, j] of E) if (q[i * 3 + 2] > 0.3 && q[j * 3 + 2] > 0.3) {
                                    g.beginPath(); g.moveTo(q[i * 3] * k, q[i * 3 + 1] * k); g.lineTo(q[j * 3] * k, q[j * 3 + 1] * k); g.stroke()
                                }
                                for (let i = 0; i < 17; i++) if (q[i * 3 + 2] > 0.3) { g.beginPath(); g.arc(q[i * 3] * k, q[i * 3 + 1] * k, 2.5, 0, 2 * Math.PI); g.fill() }
                                if (p.head) { g.beginPath(); g.arc(p.head[0] * k, p.head[1] * k, p.head[2] * k * 0.6, 0, 2 * Math.PI); g.stroke() }
                                g.font = "11px sans-serif"; g.fillText(String(p.facing), (q[0] || q[15]) * k, ((q[1] || q[16]) - 10) * k)
                            }
                        }
                    }

                    // the box being dragged out (review.html: drag = add or correct a cover)
                    DashBox {
                        id: rubber
                        objectName: "rubber-band"
                        visible: false
                        stroke: Theme.manual
                        lineWidth: 2
                        dash: [6, 3]
                    }

                    // the project editor's mouse: click selects, drag draws a box (or moves a hand-made one)
                    MouseArea {
                        id: mouse
                        objectName: "stage-mouse"
                        anchors.fill: parent
                        enabled: page.proj
                        cursorShape: Qt.CrossCursor
                        property point a
                        property bool moving: false
                        property string movePiece: ""
                        function hitManual(x, y) {
                            const bs = page.vw.boxes || []
                            for (let i = bs.length - 1; i >= 0; i--) {
                                const b = bs[i]
                                if (b.manual && b.sel && Math.abs(x - b.x) <= b.s / 2 && Math.abs(y - b.y) <= b.s / 2) return b.piece
                            }
                            return ""
                        }
                        onPressed: (m) => {
                            a = Qt.point(m.x / video.k, m.y / video.k)
                            moving = false
                            movePiece = hitManual(a.x, a.y)
                        }
                        onPositionChanged: (m) => {
                            const b = Qt.point(m.x / video.k, m.y / video.k)
                            if (!moving && Math.hypot(b.x - a.x, b.y - a.y) * video.k > 6) moving = true
                            if (!moving) return
                            if (movePiece) {  // a hand-made cover follows the mouse
                                const sel = (page.vw.boxes || []).find((x) => x.piece === movePiece)
                                if (!sel) return
                                rubber.x = (sel.x - sel.s / 2 + b.x - a.x) * video.k; rubber.y = (sel.y - sel.s / 2 + b.y - a.y) * video.k
                                rubber.width = rubber.height = sel.s * video.k
                            } else {
                                const s = Math.max(Math.abs(b.x - a.x), Math.abs(b.y - a.y))
                                rubber.x = ((a.x + b.x) / 2 - s / 2) * video.k; rubber.y = ((a.y + b.y) / 2 - s / 2) * video.k
                                rubber.width = rubber.height = s * video.k
                            }
                            rubber.visible = true
                        }
                        onReleased: (m) => {
                            const b = Qt.point(m.x / video.k, m.y / video.k)
                            rubber.visible = false
                            if (!moving) return store.call("editor_click", { x: a.x, y: a.y })
                            if (movePiece) store.call("editor_move", { piece: movePiece, dx: b.x - a.x, dy: b.y - a.y })
                            else store.call("editor_draw", { x0: a.x, y0: a.y, x1: b.x, y1: b.y })
                        }
                    }

                    // the PoC dump editor: the covers' circles, to select and drag
                    Repeater {
                        model: page.proj ? 0 : page.ed.covers.length
                        delegate: Item {
                            id: cov
                            required property int index
                            readonly property var modelData: page.ed.covers[index] || { i: index, x: 0, y: 0, r: 0 }
                            readonly property bool sel: page.ed.selected === modelData.i
                            objectName: "cover-" + modelData.i
                            x: (modelData.x - modelData.r) * video.k
                            y: (modelData.y - modelData.r) * video.k
                            width: 2 * modelData.r * video.k
                            height: width
                            Rectangle {
                                anchors.fill: parent
                                radius: width / 2
                                color: cov.sel ? Theme.accentTint : "transparent"
                                border.width: cov.sel ? 2 : 1
                                border.color: cov.sel ? Theme.accent : (drag.containsMouse ? Theme.soft : "#80FFFFFF")
                            }
                            MouseArea {
                                id: drag
                                anchors.fill: parent
                                hoverEnabled: true
                                property point last
                                cursorShape: pressed ? Qt.ClosedHandCursor : Qt.OpenHandCursor
                                onPressed: (m) => { store.call("select", { i: cov.modelData.i }); last = mapToItem(video, m.x, m.y) }
                                onPositionChanged: (m) => {
                                    if (!pressed) return
                                    const p = mapToItem(video, m.x, m.y)
                                    store.call("move_cover", { i: cov.modelData.i, dx: (p.x - last.x) / video.k, dy: (p.y - last.y) / video.k })
                                    last = p
                                }
                            }
                        }
                    }
                }

                Text {
                    objectName: "render-error"
                    visible: video.renderError.length > 0
                    anchors.bottom: parent.bottom
                    anchors.horizontalCenter: parent.horizontalCenter
                    anchors.bottomMargin: 10
                    text: video.renderError
                    color: Theme.bad
                    font.family: Theme.body; font.pixelSize: Theme.tiny
                }

                DropArea {
                    objectName: "drop-area"
                    anchors.fill: parent
                    enabled: !page.proj
                    keys: ["text/uri-list"]
                    onDropped: (drop) => { if (drop.hasUrls) { store.call("open", { video: drop.urls[0].toString(), data: page.ed.data }); drop.accept() } }
                }
            }

            // transport: play, a frame either way, where we are; the legend
            RowLayout {
                Layout.fillWidth: true
                spacing: 14
                visible: !(page.proj && page.vw.photo)
                AbstractButton {
                    objectName: "play-button"
                    implicitWidth: 40; implicitHeight: 40
                    enabled: page.vid.ready
                    onClicked: store.call("toggle_play")
                    background: Rectangle { radius: 20; color: Theme.accent }
                    contentItem: Item { Icon { anchors.centerIn: parent; name: page.ed.playing ? "pause" : "play"; color: Theme.ink; size: 18 } }
                    ToolTip.visible: hovered; ToolTip.delay: 600; ToolTip.text: "Space"
                }
                Row {
                    MmButton { objectName: "step-back"; variant: "icon"; implicitWidth: 24; implicitHeight: 28; iconName: "chevron-left"; iconSize: 18; tip: T.t("edit.back_frame"); enabled: page.vid.ready; onClicked: store.call("step", { d: -1 }) }
                    MmButton { objectName: "step-forward"; variant: "icon"; implicitWidth: 24; implicitHeight: 28; iconName: "chevron-right"; iconSize: 18; tip: T.t("edit.forward_frame"); enabled: page.vid.ready; onClicked: store.call("step", { d: 1 }) }
                }
                Text {
                    objectName: "frame-label"
                    text: page.proj ? (page.vw.frame_info || "") : (page.ed.shown < 0 ? "–" : page.ed.shown) + " / " + Math.max(0, page.vid.frames - 1)
                    font.family: Theme.body; font.pixelSize: 13; font.weight: Font.DemiBold; font.features: { "tnum": 1 }
                    color: Theme.text
                }
                // playback that can't keep up skips frames: say so rather than stutter silently
                Text {
                    objectName: "slow-play"
                    readonly property real full: page.vid.fps || 0
                    visible: page.ed.playing === true && page.ed.play_fps > 0 && page.ed.play_fps < 0.9 * full
                    text: T.t("edit.slow_play", { fps: Math.round(page.ed.play_fps), full: Math.round(full) })
                    font.family: Theme.body; font.pixelSize: 13; font.features: { "tnum": 1 }
                    color: Theme.dim
                }
                Text {
                    objectName: "frame-note"
                    visible: page.proj
                    text: page.vw.frame_note || ""
                    font.family: Theme.body; font.pixelSize: 13; font.features: { "tnum": 1 }
                    color: Theme.dim
                    elide: Text.ElideRight
                    Layout.fillWidth: true
                }
                Item { Layout.fillWidth: true; visible: !page.proj }
                Row {
                    visible: page.proj
                    spacing: 16
                    Repeater {
                        model: [["obs", "edit.legend.detected"], ["coast", "edit.legend.held"], ["manual", "edit.legend.manual"]]
                        delegate: Row {
                            required property var modelData
                            spacing: 6
                            Rectangle { width: 10; height: 10; radius: 2; color: "transparent"; border.width: 2; border.color: page.kindColour(modelData[0]); anchors.verticalCenter: parent.verticalCenter }
                            Text { text: T.t(modelData[1]); font.family: Theme.body; font.pixelSize: 12; color: Theme.dim }
                        }
                    }
                }
            }

            // the timeline: one small box per covered face, stacked, per column; click or drag to move through the video
            Rectangle {
                objectName: "timeline"
                visible: !(page.proj && page.vw.photo)
                Layout.fillWidth: true
                implicitHeight: page.proj ? 82 : 40
                radius: 10
                color: Theme.panel
                border.color: Theme.line
                Canvas {
                    id: tl
                    visible: page.proj
                    anchors.fill: parent
                    anchors.margins: 11
                    readonly property var cols: page.vw.timeline ? page.vw.timeline.cols : []
                    readonly property int tmax: page.vw.timeline ? page.vw.timeline.max : 4
                    readonly property int cur: page.vw.t || 0
                    onColsChanged: requestPaint()
                    onCurChanged: requestPaint()
                    onWidthChanged: requestPaint()
                    onPaint: {
                        const g = getContext("2d"), W = width, H = height
                        g.clearRect(0, 0, W, H)
                        const n = Math.max(1, page.vw.n || 1), c = cols.length
                        if (!c) return
                        const cw = W / c, pad = 3, gap = 1, bh = (H - 2 * pad) / Math.max(4, tmax)
                        const tint = { obs: "#803DDC84", coast: "#B3FFB020", manual: "#B329B6F6" }
                        for (let i = 0; i < c; i++) for (let k = 0; k < cols[i].length; k++) {
                            g.fillStyle = tint[cols[i][k]] || tint.obs
                            g.fillRect(i * cw + gap / 2, H - pad - (k + 1) * bh + gap / 2, Math.max(1, cw - gap), Math.max(1, bh - gap))
                        }
                        g.fillStyle = "#FFD23F"; g.fillRect((cur + 0.5) * (W / n) - 1, 0, 2, H)
                    }
                    MouseArea {
                        anchors.fill: parent
                        function to(m) { store.call("seek", { t: Math.round(Math.max(0, Math.min(1, m.x / width)) * ((page.vw.n || 1) - 1)) }) }
                        onPressed: (m) => to(m)
                        onPositionChanged: (m) => { if (pressed) to(m) }
                        ToolTip.visible: containsMouse; ToolTip.delay: 800; ToolTip.text: T.t("edit.timeline_tip")
                        hoverEnabled: true
                    }
                }
                TimeSlider {
                    visible: !page.proj
                    anchors.fill: parent
                    anchors.margins: 8
                    from: 0
                    to: Math.max(0, page.vid.frames - 1)
                    stepSize: 1
                    snapMode: Slider.SnapAlways
                    enabled: page.vid.ready
                    value: page.ed.t
                    onMoved: store.call("seek", { t: Math.round(value) })
                }
            }
        }

        // ---------------------------------------------------------------- the panel
        Rectangle {
            objectName: "editor-panel"
            visible: page.proj
            Layout.preferredWidth: 340
            Layout.fillHeight: true
            color: Theme.panel
            Rectangle { width: 1; height: parent.height; color: Theme.line }
            Flickable {
                anchors.fill: parent
                anchors.leftMargin: 1
                contentHeight: panel.implicitHeight
                clip: true
                boundsBehavior: Flickable.StopAtBounds
                ScrollBar.vertical: ScrollBar {}
                ColumnLayout {
                    id: panel
                    width: parent.width
                    spacing: 0

                    // what covers the faces
                    PanelSection {
                        title: T.t("edit.cover_with")
                        SegTabs {
                            objectName: "style-pick"
                            Layout.fillWidth: true
                            fill: true
                            accentOn: true
                            keys: (page.vw.styles || []).map((x) => x.k)
                            labels: (page.vw.styles || []).map((x) => x.label)
                            tips: (page.vw.styles || []).map((x) => x.tip)
                            current: page.vw.style || "emoji"
                            onPicked: (k) => { if (k !== page.vw.style) store.call("editor_style", { style: k }) }
                        }
                        RowLayout {
                            visible: page.vw.portrait === true
                            spacing: 10
                            Image { source: page.vw.portrait ? page.vw.portrait_pic : ""; cache: false; sourceSize: Qt.size(56 * Screen.devicePixelRatio, 56 * Screen.devicePixelRatio); smooth: true; Layout.preferredWidth: 56; Layout.preferredHeight: 56; fillMode: Image.PreserveAspectFit }
                            ColumnLayout {
                                Layout.fillWidth: true
                                MmButton { objectName: "portrait-pick"; small: true; text: T.t("edit.portrait.choose"); onClicked: photoDialog.open() }
                                Text { text: T.t("edit.portrait.hint"); wrapMode: Text.WordWrap; Layout.fillWidth: true; font.family: Theme.body; font.pixelSize: 11; color: Theme.dim }
                            }
                        }
                    }

                    // the selected box
                    PanelSection {
                        id: selSec
                        objectName: "selection"
                        title: T.t("edit.selection")
                        readonly property var sel: page.vw.sel
                        Text {
                            visible: !selSec.sel
                            text: T.t("edit.sel.none")
                            font.family: Theme.body; font.pixelSize: 13; color: Theme.dim
                            wrapMode: Text.WordWrap; Layout.fillWidth: true
                        }
                        Loader {
                            Layout.fillWidth: true
                            active: !!page.vw.sel
                            sourceComponent: selComp
                        }
                    }

                    // the emoji preset (emoji only)
                    PanelSection {
                        id: esSec
                        objectName: "emojiset"
                        visible: page.vw.style === "emoji"
                        title: T.t("emojiset.title")
                        readonly property var es: page.vw.emojiset || ({ options: [], chips: [] })
                        ComboBox {
                            id: presetBox
                            objectName: "preset-pick"
                            Layout.fillWidth: true
                            model: esSec.es.options
                            textRole: "label"
                            valueRole: "key"
                            currentIndex: Math.max(0, (esSec.es.options || []).findIndex((o) => o.key === esSec.es.current))
                            onActivated: (i) => store.call("editor_palette", { key: esSec.es.options[i].key })
                        }
                        Text { text: esSec.es.hint || ""; font.family: Theme.body; font.pixelSize: 12; color: Theme.dim; wrapMode: Text.WordWrap; Layout.fillWidth: true }
                        Flow {
                            objectName: "preset-chips"
                            Layout.fillWidth: true
                            spacing: 8
                            Repeater {
                                model: esSec.es.chips || []
                                delegate: Text {
                                    required property string modelData
                                    required property int index
                                    text: modelData
                                    font.pixelSize: 23
                                    MouseArea {
                                        anchors.fill: parent
                                        enabled: page.vw.emojiset && page.vw.emojiset.custom
                                        cursorShape: enabled ? Qt.PointingHandCursor : Qt.ArrowCursor
                                        onClicked: store.call("editor_palette_drop", { i: index })
                                        ToolTip.visible: enabled && containsMouse; ToolTip.text: T.t("emojiset.drop_tip"); hoverEnabled: true
                                    }
                                }
                            }
                        }
                        TextField {
                            id: addField
                            objectName: "preset-add"
                            visible: esSec.es.custom === true
                            Layout.fillWidth: true
                            placeholderText: T.t("emojiset.add_placeholder")
                                color: Theme.text; placeholderTextColor: Theme.faint
                                font.family: Theme.body; font.pixelSize: 13
                                background: Rectangle { radius: 8; color: Theme.deep; border.color: parent.activeFocus ? Theme.accent : Theme.line }
                            onAccepted: { store.call("editor_palette_add", { text: text }); text = "" }
                        }
                        Text { visible: addField.visible; text: T.t("emojiset.drop_hint"); font.family: Theme.body; font.pixelSize: 11; color: Theme.dim }
                        Row {
                            visible: esSec.es.overrides > 0
                            spacing: 6
                            Text { text: esSec.es.overrides_text || ""; font.family: Theme.body; font.pixelSize: 12; color: Theme.dim }
                            Text {
                                objectName: "preset-clear"
                                text: T.t("emojiset.clear"); font.family: Theme.body; font.pixelSize: 12; color: Theme.accent
                                MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: store.call("editor_clear_picks") }
                            }
                        }
                        Check {
                            objectName: "head-pose"
                            text: T.t("emojiset.pose")
                            checked: esSec.es.head_pose === true
                            onClicked: store.call("editor_head_pose", { on: checked })
                            ToolTip.visible: hovered; ToolTip.delay: 600; ToolTip.text: T.t("emojiset.pose_tip")
                        }
                    }

                    // lighting (emoji and portraits)
                    PanelSection {
                        id: ltSec
                        objectName: "lighting"
                        visible: page.vw.light ? page.vw.light.shown : false
                        title: T.t("light.title")
                        status: page.vw.light ? page.vw.light.status : ""
                        readonly property var lt: page.vw.light || ({ sliders: [] })
                        opacity: lt.enabled ? 1 : 0.75
                        RowLayout {
                            Layout.fillWidth: true
                            Text { text: T.t("light.enabled"); font.family: Theme.body; font.pixelSize: 13; font.weight: Font.DemiBold; color: Theme.text; Layout.fillWidth: true }
                            Toggle {
                                objectName: "light-enabled"
                                checked: ltSec.lt.enabled === true
                                onClicked: store.call("editor_light", { key: "enabled", value: checked })
                            }
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            Text { text: T.t("light.light_colour"); font.family: Theme.body; font.pixelSize: 13; color: Theme.soft; Layout.fillWidth: true }
                            SegTabs {
                                objectName: "light-colour"
                                accentOn: true
                                keys: ["follow", "fixed"]
                                labels: [T.t("light.follow"), T.t("light.fixed")]
                                current: ltSec.lt.fixed ? "fixed" : "follow"
                                onPicked: (k) => {
                                    if (k === "follow") store.call("editor_light", { key: "light_colour", value: null })
                                    else if (!ltSec.lt.fixed) store.call("editor_light", { key: "light_colour", value: [30, 0.0] })
                                }
                            }
                            Rectangle {
                                visible: !ltSec.lt.fixed
                                width: 16; height: 16; radius: 8
                                color: ltSec.lt.swatch || "#808080"
                                border.color: "#66FFFFFF"
                            }
                        }
                        Canvas {
                            id: wheel
                            objectName: "light-wheel"
                            visible: ltSec.lt.fixed === true
                            Layout.alignment: Qt.AlignRight
                            Layout.preferredWidth: 96; Layout.preferredHeight: 96
                            readonly property var lc: ltSec.lt.colour
                            onLcChanged: requestPaint()
                            onPaint: {
                                const g = getContext("2d"), R = 44, C = 48
                                g.clearRect(0, 0, 96, 96)
                                for (let a = 0; a < 360; a += 2) {
                                    g.beginPath(); g.moveTo(C, C); g.arc(C, C, R, (a - 1.5) * Math.PI / 180, (a + 1.5) * Math.PI / 180); g.closePath()
                                    g.fillStyle = Qt.hsla(a / 360, 1, 0.5, 1); g.fill()
                                }
                                const rg = g.createRadialGradient(C, C, 0, C, C, R)
                                rg.addColorStop(0, "rgba(235,235,235,1)"); rg.addColorStop(1, "rgba(235,235,235,0)")
                                g.fillStyle = rg; g.beginPath(); g.arc(C, C, R, 0, 2 * Math.PI); g.fill()
                                if (lc) {
                                    const x = C + Math.cos(lc[0] * Math.PI / 180) * lc[1] * R, y = C + Math.sin(lc[0] * Math.PI / 180) * lc[1] * R
                                    g.lineWidth = 2; g.strokeStyle = "#000"; g.beginPath(); g.arc(x, y, 5, 0, 2 * Math.PI); g.stroke()
                                    g.strokeStyle = "#fff"; g.lineWidth = 1.5; g.beginPath(); g.arc(x, y, 5, 0, 2 * Math.PI); g.stroke()
                                }
                            }
                            MouseArea {
                                anchors.fill: parent
                                function pick(m, live) {
                                    const dx = m.x - 48, dy = m.y - 48
                                    const a = Math.min(Math.hypot(dx, dy) / 44, 1), h = (Math.atan2(dy, dx) * 180 / Math.PI + 360) % 360
                                    store.call("editor_light", { key: "light_colour", value: [Math.round(h), Math.round(a * 100) / 100], live: live })
                                }
                                onPressed: (m) => pick(m, true)
                                onPositionChanged: (m) => { if (pressed) pick(m, true) }
                                onReleased: store.call("editor_light_end")
                                onDoubleClicked: store.call("editor_light", { key: "light_colour", value: [0, 0] })
                            }
                        }
                        GridLayout {
                            columns: 3
                            columnSpacing: 10
                            rowSpacing: 6
                            Layout.fillWidth: true
                            Repeater {
                                model: (ltSec.lt.sliders || []).length * 3
                                delegate: Loader {
                                    required property int index
                                    readonly property var sl: page.vw.light.sliders[Math.floor(index / 3)]
                                    readonly property int part: index % 3
                                    Layout.fillWidth: part === 1
                                    Layout.preferredWidth: part === 0 ? 78 : part === 2 ? 38 : -1
                                    sourceComponent: part === 0 ? slName : part === 1 ? slBar : slVal
                                }
                            }
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            Check {
                                objectName: "light-grain"
                                text: T.t("light.grain")
                                checked: ltSec.lt.grain === true
                                onClicked: store.call("editor_light", { key: "grain", value: checked })
                            }
                            Item { Layout.fillWidth: true }
                            MmButton { objectName: "light-reset"; small: true; text: T.t("light.reset"); onClicked: store.call("editor_light_reset") }
                        }
                    }

                    // keyboard shortcuts
                    PanelSection {
                        MmButton {
                            objectName: "keys-button"
                            variant: "link"
                            iconName: "keyboard"; iconSize: 15
                            text: T.t("edit.keys.title") + "   ?"
                            onClicked: store.call("editor_toggle", { what: "keys" })
                        }
                        GridLayout {
                            visible: page.vw.keys_open === true
                            columns: 2
                            columnSpacing: 12
                            rowSpacing: 2
                            Repeater {
                                model: [["← →", "edit.keys.frame"], ["⇧← →", "±10"], ["Space", "edit.keys.play"], ["edit.keys.drag", "edit.keys.drag_what"],
                                        ["edit.keys.click", "edit.keys.click_what"], ["Del", "edit.keys.del"], ["S", "edit.keys.split"], ["R", "edit.keys.why"],
                                        ["[ ]", "edit.keys.trim"], ["K", "edit.keys.undo_correction"], ["E", "edit.keys.emoji"], ["U", "edit.keys.show"],
                                        ["V", "edit.keys.view"], ["B", "edit.keys.pose"], ["D", "edit.keys.depth"], ["Esc", "edit.keys.deselect"],
                                        ["⌘Z", "edit.keys.undo"], ["⇧⌘Z", "edit.keys.redo"]]
                                delegate: Text {
                                    required property var modelData
                                    textFormat: Text.StyledText
                                    text: "<b><font color='#FFFFFF'>" + (modelData[0].startsWith("edit.") ? T.t(modelData[0]) : modelData[0]) + "</font></b> " + (modelData[1].startsWith("edit.") ? T.t(modelData[1]) : modelData[1])
                                    font.family: Theme.body; font.pixelSize: 12; color: Theme.dim
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Component { id: slName; Text { text: parent.sl.label; font.family: Theme.body; font.pixelSize: 13; color: Theme.soft
        MouseArea { anchors.fill: parent; hoverEnabled: true; ToolTip.visible: containsMouse; ToolTip.delay: 400
            ToolTip.text: "<b>" + parent.parent.sl.tip.title + "</b><br>" + parent.parent.sl.tip.what + "<br><i>" + parent.parent.sl.tip.ends + "</i>" } } }
    Component {
        id: slBar
        TimeSlider {
            objectName: "light-" + parent.sl.k
            from: parent.sl.lo; to: parent.sl.hi; stepSize: parent.sl.step
            value: parent.sl.value
            onMoved: store.call("editor_light", { key: parent.sl.k, value: Math.round(value / parent.sl.step) * parent.sl.step, live: true })
            onPressedChanged: if (!pressed) store.call("editor_light_end")
        }
    }
    Component { id: slVal; Text { text: parent.sl.text; horizontalAlignment: Text.AlignRight; font.family: Theme.body; font.pixelSize: 12; font.features: { "tnum": 1 }; color: Theme.dim } }

    // the Selection block (review.html renderSelApp)
    Component {
        id: selComp
        ColumnLayout {
            readonly property var s: page.vw.sel
            spacing: 8
            Text {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                textFormat: Text.StyledText
                text: (s.hidden ? "" : "<font size='5'>" + s.emoji + "</font> ") + "<b>" + s.who + "</b> · " + s.box + (s.by_hand ? " " + s.by_hand : "") + " · " + s.span
                font.family: Theme.body; font.pixelSize: 13; color: Theme.text
            }
            RowLayout {
                spacing: 6
                SegTabs {
                    objectName: "sel-face"
                    keys: ["face", "notface"]
                    labels: [T.t("edit.sel.their_face"), s.not_real ? T.t("edit.sel.not_real") : T.t("edit.sel.not_face")]
                    tips: [T.t("edit.sel.their_face_tip", { who: s.who }), T.t("edit.sel.not_face_tip", { who: s.who })]
                    current: s.hidden ? "notface" : "face"
                    accentOn: true
                    onPicked: (k) => store.call("editor_act", { what: k === "face" ? "restore" : "notface" })
                }
                Text { text: T.t("edit.sel.del_flips"); font.family: Theme.body; font.pixelSize: 11; color: Theme.dim }
            }
            Text { visible: s.hidden; text: s.no_emoji; wrapMode: Text.WordWrap; Layout.fillWidth: true; font.family: Theme.body; font.pixelSize: 13; color: Theme.dim }
            Flow {
                Layout.fillWidth: true
                spacing: 6
                MmButton { objectName: "sel-split"; small: true; text: T.t("edit.sel.split"); tip: s.hidden ? T.t("edit.sel.split_tip") : T.t("edit.sel.split_tip_answer"); onClicked: store.call("editor_act", { what: "split" }) }
                MmButton { objectName: "sel-trim-start"; visible: !s.hidden; small: true; text: T.t("edit.sel.trim_start"); tip: T.t("edit.sel.trim_start_tip"); onClicked: store.call("editor_act", { what: "trim_start" }) }
                MmButton { objectName: "sel-trim-end"; visible: !s.hidden; small: true; text: T.t("edit.sel.trim_end"); tip: T.t("edit.sel.trim_end_tip"); onClicked: store.call("editor_act", { what: "trim_end" }) }
                MmButton { objectName: "sel-emoji"; visible: !s.hidden; small: true; text: T.t("edit.sel.emoji"); onClicked: store.call("editor_act", { what: "emoji" }) }
                MmButton { objectName: "sel-show"; visible: !s.hidden; small: true; text: s.shown ? T.t("edit.sel.mask_again") : T.t("edit.sel.unmask"); tip: "U"; onClicked: store.call("editor_act", { what: "toggle_show" }) }
            }
            Row {
                visible: !s.hidden && !!s.trim
                spacing: 4
                Text { text: (s.trim || "") + " ·"; font.family: Theme.body; font.pixelSize: 13; color: Theme.dim }
                Text {
                    objectName: "sel-restore-length"
                    text: T.t("edit.trim.restore"); font.family: Theme.body; font.pixelSize: 13; color: Theme.accent
                    MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: store.call("editor_act", { what: "restore_length" }) }
                }
            }
            Flow {
                visible: !s.hidden && !!s.corr
                Layout.fillWidth: true
                spacing: 4
                Text { text: s.corr ? s.corr.prefix : ""; font.family: Theme.body; font.pixelSize: 13; color: Theme.dim }
                Repeater {
                    model: s.corr ? s.corr.times : []
                    delegate: Text {
                        required property var modelData
                        text: modelData.text; font.family: Theme.body; font.pixelSize: 13; color: Theme.accent
                        MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: store.call("seek", { t: modelData.t }) }
                    }
                }
                MmButton { objectName: "sel-undo-correction"; visible: s.corr ? s.corr.can_undo : false; small: true; text: T.t("edit.corr.undo"); tip: T.t("edit.corr.undo_tip"); onClicked: store.call("editor_act", { what: "undo_correction" }) }
                Text { visible: s.corr ? s.corr.first : false; text: "· " + T.t("edit.corr.first_set"); font.family: Theme.body; font.pixelSize: 13; color: Theme.dim }
            }
            Text { visible: !s.hidden; text: s.hint; wrapMode: Text.WordWrap; Layout.fillWidth: true; font.family: Theme.body; font.pixelSize: 12; color: Theme.dim }
            // the decision records behind the box
            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.line }
            Text { text: T.t("edit.decs.title"); font.family: Theme.body; font.pixelSize: 12; font.weight: Font.Bold; color: Theme.text }
            Repeater {
                model: s.decs || []
                delegate: RowLayout {
                    required property var modelData
                    spacing: 6
                    Text { text: modelData.what; Layout.minimumWidth: 52; font.family: Theme.body; font.pixelSize: 12; color: Theme.dim }
                    Text { text: modelData.answer; font.family: Theme.body; font.pixelSize: 12; font.weight: Font.DemiBold; color: Theme.text; elide: Text.ElideRight; Layout.maximumWidth: 170 }
                    Text { text: modelData.by_text; font.family: Theme.body; font.pixelSize: 11; color: modelData.by === "you" ? Theme.accent : modelData.by === "open" ? Theme.dim : Theme.warn }
                }
            }
            Text { visible: (s.decs || []).length === 0; text: s.decs_none; font.family: Theme.body; font.pixelSize: 12; color: Theme.dim }
            Text { visible: !!s.decs_more; text: s.decs_more; font.family: Theme.body; font.pixelSize: 12; color: Theme.dim }
            MmButton { objectName: "sel-why"; small: true; text: T.t("edit.decs.open"); tip: "R"; onClicked: page.key("R") }
        }
    }
}
