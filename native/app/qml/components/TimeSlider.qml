// A slider (theme.css input[type=range]): a 4 px track filled in the accent up to the value, a white 12 px thumb.
import QtQuick
import QtQuick.Controls.Basic
import ".."

Slider {
    id: s
    implicitHeight: 22
    hoverEnabled: true
    background: Item {
        x: s.leftPadding
        y: s.topPadding + s.availableHeight / 2 - 2
        width: s.availableWidth
        height: 4
        Rectangle { anchors.fill: parent; radius: 2; color: Theme.track }
        Rectangle { width: s.visualPosition * parent.width; height: parent.height; radius: 2; color: Theme.accent }
    }
    handle: Rectangle {
        x: s.leftPadding + s.visualPosition * (s.availableWidth - width)
        y: s.topPadding + s.availableHeight / 2 - height / 2
        width: s.pressed || s.hovered ? 14 : 12
        height: width
        radius: width / 2
        color: "#FFFFFF"
    }
}
