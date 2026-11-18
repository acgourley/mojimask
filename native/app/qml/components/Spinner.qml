// theme.css .spin: a ring with an accent arc, turning.
import QtQuick
import ".."

Item {
    id: s
    property int size: 30
    property real thick: size < 20 ? 2 : 3
    width: size
    height: size
    Rectangle { anchors.fill: parent; radius: width / 2; color: "transparent"; border.color: Theme.line; border.width: s.thick }
    Canvas {
        id: arc
        anchors.fill: parent
        onPaint: {
            const g = getContext("2d")
            g.reset()
            g.strokeStyle = Theme.accent
            g.lineWidth = s.thick
            g.beginPath()
            g.arc(width / 2, height / 2, width / 2 - s.thick / 2, -Math.PI * 0.75, -Math.PI * 0.25)
            g.stroke()
        }
        RotationAnimation on rotation { from: 0; to: 360; duration: 900; loops: Animation.Infinite; running: s.visible }
    }
}
