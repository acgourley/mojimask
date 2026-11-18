// A section of the Editor's side panel (review.html aside section): an uppercase heading, then its content.
import QtQuick
import QtQuick.Layouts
import ".."

Rectangle {
    id: sec
    property string title: ""
    property string status: ""
    default property alias content: col.data
    color: "transparent"
    Layout.fillWidth: true
    implicitHeight: col.implicitHeight + 28
    Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Theme.line }
    ColumnLayout {
        id: col
        x: 16; y: 14
        width: parent.width - 32
        spacing: 8
        Row {
            visible: sec.title !== ""
            spacing: 6
            Text {
                text: sec.title.toUpperCase()
                font.family: Theme.body; font.pixelSize: 11; font.weight: Font.DemiBold; font.letterSpacing: 1.5
                color: Theme.faint
            }
            Text { text: sec.status; visible: sec.status !== ""; font.family: Theme.body; font.pixelSize: 11; color: Theme.dim }
        }
    }
}
