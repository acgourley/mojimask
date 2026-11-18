// Rounded corners over content that can't be masked cheaply (a live canvas, a reloading preview): four wedges in the
// background colour, drawn on top. anchors.fill the content.
import QtQuick
import QtQuick.Shapes
import ".."

Item {
    id: rc
    property real radius: 14
    property color color: Theme.bg
    Repeater {
        model: 4
        delegate: Shape {
            required property int index
            width: rc.radius; height: rc.radius
            x: index % 2 ? rc.width - rc.radius : 0
            y: index > 1 ? rc.height - rc.radius : 0
            rotation: [0, 90, 270, 180][index]
            preferredRendererType: Shape.CurveRenderer
            ShapePath {
                strokeWidth: -1
                fillColor: rc.color
                startX: 0; startY: 0
                PathLine { x: rc.radius; y: 0 }
                PathArc { x: 0; y: rc.radius; radiusX: rc.radius; radiusY: rc.radius; direction: PathArc.Counterclockwise }
                PathLine { x: 0; y: 0 }
            }
        }
    }
}
