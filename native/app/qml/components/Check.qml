// theme.css input[type=checkbox]: a 16 px box, the accent with an ink tick when checked; `text` beside it.
import QtQuick
import QtQuick.Controls.Basic
import ".."

AbstractButton {
    id: cb
    checkable: true
    hoverEnabled: true
    implicitHeight: Math.max(18, label.implicitHeight)
    implicitWidth: 16 + (text ? 9 + label.implicitWidth : 0)
    background: Item {}
    contentItem: Item {
        Rectangle {
            id: box
            width: 16; height: 16; radius: 4
            anchors.verticalCenter: parent.verticalCenter
            color: cb.checked ? Theme.accent : "transparent"
            border.width: cb.checked ? 0 : 1.5
            border.color: Theme.line2
            Text { anchors.centerIn: parent; visible: cb.checked; text: "✓"; color: Theme.ink; font.pixelSize: 12; font.weight: Font.Black }
        }
        Text {
            id: label
            x: 25
            anchors.verticalCenter: parent.verticalCenter
            text: cb.text
            font.family: Theme.body; font.pixelSize: 13
            color: Theme.soft
        }
    }
}
