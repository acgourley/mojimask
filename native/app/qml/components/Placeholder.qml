// A page not built yet in the native app: its title and what it will hold.
import QtQuick
import QtQuick.Layouts
import ".."

Rectangle {
    id: p
    property string title
    property string what
    color: Theme.bg
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 32
        spacing: 16
        PageHead { title: p.title; summary: p.what; Layout.fillWidth: true }
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            radius: Theme.radiusBig
            color: Theme.panel
            border.color: Theme.line
            Text {
                anchors.centerIn: parent
                text: qsTr("Not in the native app yet")
                font.family: Theme.body; font.pixelSize: Theme.textSize
                color: Theme.faint
            }
        }
    }
}
