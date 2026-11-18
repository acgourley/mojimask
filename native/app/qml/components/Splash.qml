// The window's opening picture (mojimask/splash.py): the app icon and the wordmark on black, over everything, then a
// fade (splash.FADE_S) to the page. The icon is drawn as packaging/make_icon.py draws the real one: The Laughing
// Cavalier (native/web/assets/icon_art.jpg) on a rounded tile, the disguise emoji over his face. The native window is
// up at once, so it holds for HOLD_MS (enough to read as the app's face, not a flash) and then fades. A drag on it
// moves
// the window, as on the Python app's splash. Shown once per launch: a live QML reload (--dev) skips it.
import QtQuick
import QtQuick.Window
import ".."

Rectangle {
    id: splash
    objectName: "splash"
    color: "#000000"
    visible: !qmlReloaded
    readonly property int holdMs: 700
    readonly property int fadeMs: 300
    // make_icon.py's numbers, as fractions of the tile (splash.py)
    readonly property real tile: 168
    readonly property real radiusF: 185.4 / 824
    readonly property real emojiX: 165 / 824
    readonly property real emojiY: 189 / 824
    readonly property real emojiSize: 402 / 824
    readonly property real emojiTurn: 11.17

    // the page behind gets no clicks meanwhile; a drag moves the window (a MouseArea here would take the press
    // before a DragHandler on this item could see it)
    MouseArea {
        anchors.fill: parent
        hoverEnabled: true
        onPressed: splash.Window.window.startSystemMove()
    }

    Column {
        id: face
        anchors.centerIn: parent
        spacing: 28
        Item {
            objectName: "splash-icon"
            anchors.horizontalCenter: parent.horizontalCenter
            width: splash.tile; height: splash.tile
            clip: true
            Image {
                anchors.fill: parent
                source: webDir + "assets/icon_art.jpg"
                fillMode: Image.PreserveAspectCrop
                sourceSize: Qt.size(width * Screen.devicePixelRatio, height * Screen.devicePixelRatio)   // decoded at the size drawn (Picture.qml)
                smooth: true
            }
            Text {
                text: "🥸"
                font.family: "Apple Color Emoji"
                font.pixelSize: splash.tile * splash.emojiSize
                width: splash.tile * splash.emojiSize; height: width
                horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter
                x: splash.tile * splash.emojiX; y: splash.tile * splash.emojiY
                transformOrigin: Item.TopLeft
                rotation: -splash.emojiTurn
                antialiasing: true
                layer.enabled: true; layer.smooth: true   // a rotated glyph bitmap is sampled smoothly only as a layer
            }
            RoundCorners { anchors.fill: parent; radius: splash.tile * splash.radiusF; color: splash.color }
        }
        Wordmark { objectName: "splash-word"; size: 56; anchors.horizontalCenter: parent.horizontalCenter }
    }

    Timer { running: splash.visible; interval: splash.holdMs; onTriggered: splash.state = "out" }
    states: State { name: "out"; PropertyChanges { target: splash; opacity: 0 } }
    transitions: Transition {
        to: "out"
        SequentialAnimation {
            NumberAnimation { property: "opacity"; duration: splash.fadeMs; easing.type: Easing.InQuad }
            PropertyAction { target: splash; property: "visible"; value: false }
        }
    }
}
