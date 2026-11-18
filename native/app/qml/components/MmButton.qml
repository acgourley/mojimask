// A button in theme.css's styles: "plain" (outlined), "primary" (accent), "raised", "icon" (square, borderless),
// "link", "danger" (dialog_ui.js's red OK). `big` is .big (a bigger, bolder button; outlined ones get the brighter
// line); `small` is .small; `iconName` puts an icons.json icon before the text; `trailingIcon` after it.
import QtQuick
import QtQuick.Controls.Basic
import ".."

Button {
    id: b
    property string variant: "plain"
    property bool big: false
    property bool small: false
    property string iconName: ""
    property string trailingIcon: ""
    property int iconSize: big ? 18 : 16
    property string tip: ""
    property real radius: big ? 12 : small ? 8 : Theme.radius
    readonly property bool hot: hovered && enabled
    focusPolicy: Qt.TabFocus
    hoverEnabled: true
    padding: variant === "icon" || variant === "link" ? 0 : (big ? 13 : small ? 5 : 8)
    leftPadding: variant === "icon" || variant === "link" ? 0 : (big ? 20 : 12)
    rightPadding: leftPadding
    implicitHeight: variant === "icon" ? 34 : contentItem.implicitHeight + topPadding + bottomPadding
    implicitWidth: variant === "icon" ? 34 : contentItem.implicitWidth + leftPadding + rightPadding
    font.family: Theme.body
    font.pixelSize: big ? 15 : Theme.small
    font.weight: variant === "primary" || variant === "danger" ? Font.Bold : (variant === "raised" || big) ? Font.DemiBold : Font.Medium
    opacity: enabled ? 1 : 0.45
    ToolTip.visible: hovered && tip !== ""
    ToolTip.delay: 600
    ToolTip.text: tip

    readonly property color fg: b.variant === "primary" ? Theme.ink
                               : b.variant === "danger" ? "#1A0505"
                               : b.variant === "raised" ? Theme.text
                               : b.variant === "link" ? (b.hot ? Theme.accent : Theme.soft)
                               : b.variant === "icon" ? (b.hot ? Theme.text : Theme.dim)
                               : (b.hot || b.big ? Theme.text : Theme.soft)

    contentItem: Row {
        spacing: b.big ? 10 : 7
        Icon {
            visible: b.iconName !== ""
            name: b.iconName
            size: b.iconSize
            color: b.fg
            anchors.verticalCenter: parent.verticalCenter
        }
        Text {
            visible: b.text !== ""
            text: b.text
            font: b.font
            color: b.fg
            anchors.verticalCenter: parent.verticalCenter
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
        }
        Icon {
            visible: b.trailingIcon !== ""
            name: b.trailingIcon
            size: b.iconSize
            color: b.fg
            anchors.verticalCenter: parent.verticalCenter
        }
    }
    background: Rectangle {
        radius: b.radius
        visible: b.variant !== "link"
        color: b.variant === "primary" ? (b.hot ? Qt.lighter(Theme.accent, 1.06) : Theme.accent)
             : b.variant === "danger" ? Theme.bad
             : b.variant === "raised" ? (b.hot ? Theme.raised2 : Theme.raised)
             : b.variant === "icon" ? (b.hot ? Theme.raised : "transparent")
             : (b.down ? Theme.raised : "transparent")
        border.width: b.variant === "plain" ? 1 : 0
        border.color: b.hot || b.big ? Theme.line2 : Theme.line
        Rectangle { // focus ring
            anchors.fill: parent
            anchors.margins: -3
            radius: parent.radius + 3
            color: "transparent"
            border.width: 2
            border.color: Theme.accent
            visible: b.visualFocus
        }
    }
}
