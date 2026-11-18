// theme.css .devtag: the developer-mode tag.
import QtQuick
import ".."

Rectangle {
    implicitWidth: t.implicitWidth + 12
    implicitHeight: t.implicitHeight + 4
    radius: 5
    color: "transparent"
    border.color: Theme.raised2
    Text { id: t; anchors.centerIn: parent; text: "DEV"; font.family: Theme.body; font.pixelSize: 10; font.weight: Font.Bold; font.letterSpacing: 1; color: Theme.dim }
}
