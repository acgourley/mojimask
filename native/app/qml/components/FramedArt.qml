// One of the drop card's three old paintings (public domain, web/assets/gallery), in a gilt frame, its face covered
// by an emoji drawn as text (app.html .framed-art).
import QtQuick
import QtQuick.Window
import QtQuick.Effects
import ".."

Item {
    id: f
    property string file
    property int artW: 96
    property int artH: 130
    property string emoji
    property real ex
    property real ey
    property int esize
    property real erot
    width: artW + 20
    height: artH + 20
    RectangularShadow { anchors.fill: frame; offset.y: 10; blur: 20; color: "#AA000000" }
    Rectangle {
        id: frame
        anchors.fill: parent
        gradient: Gradient {
            orientation: Gradient.Horizontal
            GradientStop { position: 0; color: "#8A6238" }
            GradientStop { position: 0.33; color: "#4A311A" }
            GradientStop { position: 0.67; color: "#3A2512" }
            GradientStop { position: 1; color: "#6E4C2A" }
        }
        Rectangle {
            anchors.fill: parent; anchors.margins: 7; color: "#1A1008"
            Rectangle {
                anchors.fill: parent; anchors.margins: 1
                gradient: Gradient {
                    orientation: Gradient.Horizontal
                    GradientStop { position: 0; color: "#8C6B36" }
                    GradientStop { position: 0.5; color: "#C9A565" }
                    GradientStop { position: 1; color: "#F6E2A8" }
                }
                Item {
                    anchors.fill: parent; anchors.margins: 2
                    clip: true
                    Rectangle { anchors.fill: parent; color: "#1A1008" }
                    Image { anchors.fill: parent; source: webDir + "assets/gallery/" + f.file; fillMode: Image.PreserveAspectCrop; sourceSize: Qt.size(width * Screen.devicePixelRatio, height * Screen.devicePixelRatio); smooth: true }
                    Text {
                        text: f.emoji
                        font.family: "Apple Color Emoji"
                        font.pixelSize: f.esize
                        x: parent.width * f.ex
                        y: parent.height * f.ey
                        transformOrigin: Item.TopLeft
                        rotation: f.erot
                        antialiasing: true
                        layer.enabled: true; layer.smooth: true   // a rotated glyph bitmap is sampled smoothly only as a layer
                        opacity: 0.95
                    }
                }
            }
        }
    }
}
