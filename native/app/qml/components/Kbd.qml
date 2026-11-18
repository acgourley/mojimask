// theme.css kbd: a key, as the answers show theirs. tone "accent" (a chosen answer) or "warn" (a rule's).
import QtQuick
import ".."

Rectangle {
    property alias text: k.text
    property string tone: ""
    implicitWidth: k.implicitWidth + 12
    implicitHeight: k.implicitHeight + 2
    radius: 5
    color: "transparent"
    border.color: tone === "accent" ? "#80FFD23F" : tone === "warn" ? "#80F5A524" : "#4DFFFFFF"
    Text { id: k; anchors.centerIn: parent; font.family: Theme.body; font.pixelSize: 11; font.weight: Font.DemiBold; color: parent.tone === "accent" ? Theme.accent : parent.tone === "warn" ? Theme.warn : Theme.soft }
}
