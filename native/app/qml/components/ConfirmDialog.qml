// web/dialog_ui.js's Dialog.ask / Dialog.tell: a question with OK (danger: red) and Cancel. Enter answers OK, Esc
// cancels. ask(text, ok, danger, cancel, then): `then` runs on OK; cancel "" for a message with OK only.
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import ".."

Popup {
    id: d
    property string text
    property string okLabel: T.t("common.ok")
    property string cancelLabel: T.t("common.cancel")
    property bool danger: false
    property var then: null
    modal: true
    focus: true
    anchors.centerIn: Overlay.overlay
    width: Math.min(520, parent ? parent.width - 32 : 520)
    padding: 20
    bottomPadding: 16
    closePolicy: Popup.CloseOnEscape
    Overlay.modal: Rectangle { color: "#9907060D" }

    function ask(text, ok, danger, cancel, then) {
        d.text = text
        d.okLabel = ok || T.t("common.ok")
        d.danger = !!danger
        d.cancelLabel = cancel === undefined || cancel === null ? T.t("common.cancel") : cancel
        d.then = then || null
        open()
        okButton.forceActiveFocus()
    }
    function accept() {
        const f = d.then
        close()
        if (f) f()
    }

    background: Rectangle { color: Theme.panel; radius: 14; border.color: Theme.line }
    contentItem: ColumnLayout {
        spacing: 14
        Keys.onReturnPressed: d.accept()
        Keys.onEnterPressed: d.accept()
        Text {
            objectName: "confirm-text"
            text: d.text
            Layout.fillWidth: true
            wrapMode: Text.Wrap
            font.family: Theme.body; font.pixelSize: Theme.textSize
            lineHeight: Theme.lineHeight
            color: Theme.text
        }
        RowLayout {
            Layout.alignment: Qt.AlignRight
            spacing: 8
            MmButton { objectName: "confirm-cancel"; visible: d.cancelLabel !== ""; text: d.cancelLabel; onClicked: d.close() }
            MmButton {
                id: okButton
                objectName: "confirm-ok"
                variant: d.danger ? "danger" : "primary"
                text: d.okLabel
                onClicked: d.accept()
                Keys.onReturnPressed: d.accept()
            }
        }
    }
}
