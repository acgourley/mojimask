// A small label on a chip (theme.css .chip); `tone` "accent" for the accent tint, "dashed" for a rule's or a grade's.
import QtQuick
import ".."

Rectangle {
    property alias text: label.text
    property string tone: ""
    implicitWidth: label.implicitWidth + 20
    implicitHeight: label.implicitHeight + 8
    radius: 8
    color: tone === "accent" ? Theme.accentTint : tone === "dashed" ? "transparent" : Theme.chip
    border.width: tone === "accent" || tone === "dashed" ? 1 : 0
    border.color: tone === "dashed" ? Theme.line2 : Theme.accent
    Text {
        id: label
        anchors.centerIn: parent
        font.family: Theme.body
        font.pixelSize: Theme.tiny
        font.features: { "tnum": 1 }
        color: parent.tone === "accent" ? Theme.accent : Theme.soft
    }
}
