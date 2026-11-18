// A filter chip (theme.css .pill): "To answer 12". `on` when chosen.
import QtQuick
import QtQuick.Controls.Basic
import ".."

AbstractButton {
    id: p
    property bool on: false
    hoverEnabled: true
    padding: 6; leftPadding: 12; rightPadding: 12
    contentItem: Text {
        text: p.text
        font.family: Theme.body; font.pixelSize: Theme.small; font.weight: p.on ? Font.DemiBold : Font.Medium
        color: p.on ? Theme.accent : Theme.soft
    }
    background: Rectangle { radius: 16; color: p.on ? Theme.accentTint : Theme.panel; border.color: p.on ? Theme.accent : p.hovered ? Theme.line2 : Theme.line }
}
