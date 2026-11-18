// A page's heading row (theme.css .pagehead): title, a line under it, and room on the right for actions.
import QtQuick
import QtQuick.Layouts
import ".."

ColumnLayout {
    property alias title: t.text
    property alias summary: s.text
    default property alias actions: right.data
    spacing: 6
    RowLayout {
        Layout.fillWidth: true
        Text {
            id: t
            font.family: Theme.display; font.pixelSize: Theme.h2; font.weight: Font.ExtraBold
            color: Theme.text
        }
        Item { Layout.fillWidth: true }
        Row { id: right; spacing: Theme.gap }
    }
    Text {
        id: s
        visible: text.length > 0
        Layout.fillWidth: true
        wrapMode: Text.Wrap
        font.family: Theme.body; font.pixelSize: Theme.textSize
        color: Theme.dim
    }
}
