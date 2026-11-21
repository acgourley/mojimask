// People's pictures (people.rs Pic): frame t cropped to the rectangle (x, y, w, h) in frame pixels, scaled to this
// item; the face's square outlined in white with a dark edge (`box`), as people.html paint() draws it.
import QtQuick
import QtQuick.Window
import ".."

Picture {
    id: c
    property var pic: null
    property bool box: true
    readonly property real k: pic && pic.w > 0 ? width / pic.w : 1
    readonly property real lw: Math.max(1.5, width / 160)
    clip: true
    Rectangle { anchors.fill: parent; color: "#000000" }
    Image {
        // the frame's rectangle pic.rect, cut and scaled by the server (Picture.qml)
        source: c.pic ? c.sized(c.pic.url, Math.max(c.width, c.height) * Screen.devicePixelRatio) : ""
        x: c.pic ? (c.pic.rect[0] - c.pic.x) * c.k : 0
        y: c.pic ? (c.pic.rect[1] - c.pic.y) * c.k : 0
        width: c.pic ? c.pic.rect[2] * c.k : 0
        height: c.pic ? c.pic.rect[3] * c.k : 0
        asynchronous: true
        cache: true
        retainWhileLoading: true   // a new size (a resize) reloads it: keep the old picture until then
        smooth: true
    }
    Item {
        visible: c.box && !!c.pic
        x: c.pic ? (c.pic.face[0] - c.pic.face[2] / 2 - c.pic.x) * c.k : 0
        y: c.pic ? (c.pic.face[1] - c.pic.face[2] / 2 - c.pic.y) * c.k : 0
        width: c.pic ? c.pic.face[2] * c.k : 0
        height: width
        Rectangle { anchors.fill: parent; anchors.margins: -(c.lw + 2) / 2; color: "transparent"; border.color: "#99000000"; border.width: c.lw + 2 }
        Rectangle { anchors.fill: parent; anchors.margins: -c.lw / 2; color: "transparent"; border.color: "#FFFFFF"; border.width: c.lw }
    }
}
