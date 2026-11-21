// A crop of a scanned frame (decide.rs Crop): the square around (cx, cy), `half` frame pixels to each side, scaled to
// this item (the picture is the frame's rectangle crop.rect, cut and scaled by the server); the grey box (the person) and the green box (the piece) outlined (never covers: the Decisions page shows faces
// as they are); a progress line while it loops. Rounded like the page's canvases.
import QtQuick
import QtQuick.Window
import QtQuick.Effects
import ".."

Picture {
    id: c
    property var crop: null
    property real progress: -1
    property real radius: 8
    property bool boxes: true
    readonly property real n: width
    readonly property real k: crop && crop.half > 0 ? width / (2 * crop.half) : 1
    // the longer side to ask the server for (Picture.qml)
    readonly property real want: Math.max(width, height) * Screen.devicePixelRatio
    readonly property real lw: n < 100 ? 1.5 : n < 300 ? 2 : 3
    implicitWidth: 52
    implicitHeight: width

    Item {
        id: content
        anchors.fill: parent
        visible: false
        layer.enabled: true
        Rectangle { anchors.fill: parent; color: "#000000" }
        Image {
            id: im
            source: c.crop ? c.sized(c.crop.url, c.want) : ""
            x: c.crop ? (c.crop.rect[0] - c.crop.cx + c.crop.half) * c.k : 0
            y: c.crop ? (c.crop.rect[1] - c.crop.cy + c.crop.half) * c.k : 0
            width: c.crop ? c.crop.rect[2] * c.k : 0
            height: c.crop ? c.crop.rect[3] * c.k : 0
            asynchronous: true
            cache: true
            retainWhileLoading: true
            smooth: true
        }
        Repeater {
            // the grey box under the green one; a thin white edge outside so both show on any picture
            model: c.boxes && c.crop ? [[c.crop.person, Theme.person], [c.crop.piece, Theme.piece]] : []
            delegate: Item {
                required property var modelData
                readonly property var b: modelData[0]
                visible: !!b
                x: b ? (b[0] - c.crop.cx + c.crop.half) * c.k : 0
                y: b ? (b[1] - c.crop.cy + c.crop.half) * c.k : 0
                width: b ? (b[2] - b[0]) * c.k : 0
                height: b ? (b[3] - b[1]) * c.k : 0
                Rectangle {
                    visible: c.n >= 100
                    anchors.fill: parent
                    anchors.margins: -c.lw
                    color: "transparent"
                    border.color: "#FFFFFF"
                    border.width: 1
                }
                Rectangle {
                    anchors.fill: parent
                    anchors.margins: -c.lw / 2
                    color: "transparent"
                    border.color: modelData[1]
                    border.width: c.lw
                }
            }
        }
        Rectangle {
            visible: c.progress >= 0
            anchors.bottom: parent.bottom
            width: parent.width
            height: 3
            color: "#80000000"
            Rectangle { width: parent.width * Math.max(0, c.progress); height: 3; color: Theme.accent }
        }
    }
    Rectangle {
        id: mask
        anchors.fill: parent
        radius: c.radius
        visible: false
        layer.enabled: true
    }
    MultiEffect {
        anchors.fill: parent
        source: content
        maskEnabled: true
        maskSource: mask
        maskThresholdMin: 0.5
        maskSpreadAtMin: 1.0
    }
}
