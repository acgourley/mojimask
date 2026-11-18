// theme.css input[type=checkbox].switch: a 34 x 20 track, the accent when on, a 16 px knob.
import QtQuick
import QtQuick.Controls.Basic
import ".."

AbstractButton {
    id: sw
    checkable: true
    implicitWidth: 34
    implicitHeight: 20
    hoverEnabled: true
    background: Rectangle {
        radius: 10
        color: sw.checked ? Theme.accent : Theme.track
        Behavior on color { ColorAnimation { duration: 150 } }
        Rectangle {
            width: 16; height: 16; radius: 8
            y: 2; x: sw.checked ? 16 : 2
            color: sw.checked ? Theme.ink : Theme.soft
            Behavior on x { NumberAnimation { duration: 150 } }
        }
    }
    contentItem: Item {}
}
