// A box outline, solid or dashed (review.html strokeBox): `dash` [on, off] in px, [] for solid.
import QtQuick
import QtQuick.Shapes

Shape {
    id: b
    property color stroke: "white"
    property real lineWidth: 1.5
    property var dash: []
    preferredRendererType: Shape.CurveRenderer
    ShapePath {
        strokeColor: b.stroke
        strokeWidth: b.lineWidth
        fillColor: "transparent"
        strokeStyle: b.dash.length ? ShapePath.DashLine : ShapePath.SolidLine
        dashPattern: b.dash.length ? [b.dash[0] / b.lineWidth, b.dash[1] / b.lineWidth] : [4, 2]
        startX: 0; startY: 0
        PathLine { x: b.width; y: 0 }
        PathLine { x: b.width; y: b.height }
        PathLine { x: 0; y: b.height }
        PathLine { x: 0; y: 0 }
    }
}
