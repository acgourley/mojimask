// One choice of a few (theme.css .seg.tabs): the bar's steps, a view switch. `keys` and `labels` in order; `current`
// is the chosen key; `picked(key)` when a person picks one. `counts` (optional): a number on a tab (.count), -1 for
// none; `tips`: tooltips. Each tab's objectName is "<name>-<key>".
import QtQuick
import QtQuick.Controls.Basic
import ".."

Rectangle {
    id: seg
    property var keys: []
    property var labels: []
    property var counts: []
    property var tips: []
    property string current: ""
    // .seg.fill: the tabs share the width; accentOn: the chosen one in the accent (.seg button.on), goodKey's in green
    property bool fill: false
    property bool accentOn: false
    property string goodKey: ""
    signal picked(string key)
    implicitWidth: row.implicitWidth + 6
    implicitHeight: row.implicitHeight + 6
    radius: Theme.radius
    color: Theme.deep
    border.color: Theme.line
    border.width: 1

    Row {
        id: row
        anchors.centerIn: parent
        spacing: 2
        readonly property real each: seg.fill ? (seg.width - 6 - 2 * (seg.keys.length - 1)) / Math.max(1, seg.keys.length) : -1
        Repeater {
            model: seg.keys.length
            delegate: AbstractButton {
                id: tab
                required property int index
                readonly property string key: seg.keys[index]
                readonly property bool on: seg.current === key
                readonly property int count: seg.counts[index] === undefined || seg.counts[index] === null ? -1 : seg.counts[index]
                objectName: seg.objectName + "-" + key
                text: seg.labels[index] || key
                width: row.each > 0 ? row.each : implicitWidth
                hoverEnabled: true
                padding: 6
                leftPadding: 14
                rightPadding: 14
                onClicked: seg.picked(key)
                ToolTip.visible: hovered && (seg.tips[index] || "") !== ""
                ToolTip.delay: 600
                ToolTip.text: seg.tips[index] || ""
                readonly property color onColor: seg.accentOn ? (tab.key === seg.goodKey ? Theme.good : Theme.accent) : Theme.text
                contentItem: Item {
                  implicitWidth: tabRow.implicitWidth
                  implicitHeight: tabRow.implicitHeight
                  Row {
                    id: tabRow
                    anchors.centerIn: parent
                    spacing: 7
                    Text {
                        text: tab.text
                        font.family: Theme.body
                        font.pixelSize: Theme.small
                        font.weight: tab.on ? Font.DemiBold : Font.Medium
                        color: tab.on ? tab.onColor : tab.hovered ? Theme.text : Theme.dim
                        anchors.verticalCenter: parent.verticalCenter
                    }
                    Rectangle {
                        visible: tab.count >= 0
                        anchors.verticalCenter: parent.verticalCenter
                        width: cnt.implicitWidth + 12
                        height: cnt.implicitHeight + 2
                        radius: 8
                        color: tab.on ? Theme.raised2 : Theme.chip
                        Text {
                            id: cnt
                            anchors.centerIn: parent
                            text: tab.count
                            font.family: Theme.body; font.pixelSize: 11; font.weight: Font.DemiBold
                            color: tab.on ? Theme.soft : Theme.faint
                        }
                    }
                  }
                }
                background: Rectangle {
                    radius: 8
                    color: !tab.on ? "transparent" : !seg.accentOn ? Theme.raised : tab.key === seg.goodKey ? "#293DDC84" : Theme.accentSoft
                }
            }
        }
    }
}
