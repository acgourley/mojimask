// An icon from native/web/icons.js (Lucide), drawn in a colour: Icon { name: "play"; color: Theme.dim; size: 16 }.
import QtQuick
import ".."

Image {
    property string name
    property color color: Theme.dim
    property int size: 16
    width: size
    height: size
    sourceSize: Qt.size(size * 2, size * 2)
    source: name ? T.icon(name, color) : ""
    smooth: true
}
