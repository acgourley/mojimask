// Progress as a thin accent line (0..1).
import QtQuick
import ".."

Rectangle {
    property real value: 0
    implicitHeight: 4
    implicitWidth: 160
    radius: 2
    color: Theme.track
    Rectangle { width: parent.width * Math.max(0, Math.min(1, parent.value)); height: parent.height; radius: 2; color: Theme.accent }
}
