// One of the selected question's pictures (decisions.html Loops): its still, and while the question is selected it
// plays its box's own frames, first detection to last, in real time and straight round again (store.call
// "loop_frames"); the pointer over it scrubs (left to right through the frames). A box seen once stays still.
import QtQuick
import QtQuick.Controls.Basic
import ".."

Column {
    id: lp
    property var pic
    property int picIndex: 0
    property string detailKey: ""
    property bool mmPublic: false
    property real imgW: 1
    property real imgH: 1
    property var frames: []
    property var span: [0, 0]
    property real fps: 24
    property int frame: -1
    property bool scrubbing: false
    spacing: 8

    function load() {
        const r = store.call("loop_frames", { pic: picIndex })
        frame = -1
        frames = r && r.frames ? r.frames : []
        span = r && r.span ? r.span : [0, 0]
        fps = r && r.fps ? r.fps : 24
    }
    onDetailKeyChanged: Qt.callLater(load)
    Component.onCompleted: Qt.callLater(load)

    Timer {
        interval: Math.max(16, 1000 / lp.fps)
        repeat: true
        running: lp.visible && lp.frames.length > 1 && !lp.scrubbing
        onTriggered: lp.frame = (lp.frame + 1) % lp.frames.length
    }

    Crop {
        id: crop
        objectName: "crop"
        width: lp.width
        height: width
        radius: 12
        mmPublic: lp.mmPublic
        imgW: lp.imgW
        imgH: lp.imgH
        crop: lp.frame >= 0 && lp.frame < lp.frames.length ? lp.frames[lp.frame] : (lp.pic ? lp.pic.still : null)
        progress: lp.frame >= 0 && lp.frames.length > 1 ? lp.frame / (lp.frames.length - 1) : -1
        MouseArea {
            anchors.fill: parent
            hoverEnabled: true
            onPositionChanged: (m) => {
                if (lp.frames.length < 2) return
                lp.scrubbing = true
                lp.frame = Math.max(0, Math.min(lp.frames.length - 1, Math.floor(m.x / width * lp.frames.length)))
            }
            onExited: lp.scrubbing = false
            // to look closer: that moment in Edit (the one showing, if it's playing), with this picture's box selected
            cursorShape: Qt.PointingHandCursor
            onClicked: {
                const c = lp.frame >= 0 && lp.frame < lp.frames.length ? lp.frames[lp.frame] : (lp.pic ? lp.pic.still : null)
                if (c && lp.pic) store.call("edit_at", { t: c.t, piece: lp.pic.piece })
            }
            ToolTip.visible: containsMouse; ToolTip.delay: 800; ToolTip.text: T.t("decisions.pic.open_tip")
        }
    }
    Eyebrow {
        width: lp.width
        elide: Text.ElideRight
        font.features: { "tnum": 1 }
        text: !lp.pic ? "" : lp.pic.wide || lp.frame < 0 || lp.frame >= lp.frames.length ? lp.pic.cap
              : lp.pic.label + " · " + T.t("decisions.pic.frame_of", { t: lp.frames[lp.frame].t, a: lp.span[0], b: lp.span[1] })
    }
    // load every frame of the loop ahead of playing it
    Repeater {
        model: lp.frames
        delegate: Image { required property var modelData; width: 0; height: 0; visible: false; source: modelData.url; asynchronous: true; cache: true }
    }
}
