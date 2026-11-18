// The wordmark (native/web/brand.json): the name in Nunito 900, the first part white, the rest in the accent.
import QtQuick
import ".."

Row {
    property int size: 22
    spacing: 0
    Text {
        text: Theme.brandName.slice(0, Theme.brandSplit)
        font.family: Theme.display; font.pixelSize: parent.size; font.weight: Font.Black; font.letterSpacing: -0.01 * parent.size
        color: Theme.brandFirst
    }
    Text {
        text: Theme.brandName.slice(Theme.brandSplit)
        font.family: Theme.display; font.pixelSize: parent.size; font.weight: Font.Black; font.letterSpacing: -0.01 * parent.size
        color: Theme.brandSecond
    }
}
