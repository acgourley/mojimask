// People (web/people.html): everyone the scan counted as a person, one card each. Everyone starts masked; a card
// unmasks someone who's happy to be seen, names them, or picks their emoji (facts about their boxes, saved as you go).
// What's shown is store.state.people (people.rs); what's done is store.call("people_say" / "people_toggle").
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import ".."
import "../components"

Rectangle {
    id: page
    objectName: "page-people"
    color: Theme.bg
    readonly property var ppl: store.state.people
    readonly property var proj: store.state.project

    ScrollView {
        id: scroll
        anchors.fill: parent
        contentWidth: availableWidth
        ColumnLayout {
            x: 32
            width: scroll.availableWidth - 64
            spacing: 20
            Item { Layout.preferredHeight: 4 }
            RowLayout {
                Layout.fillWidth: true
                spacing: 24
                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 6
                    Row {
                        spacing: 14
                        Text { id: h1; text: T.t("nav.people"); font.family: Theme.display; font.pixelSize: Theme.h1; font.weight: Font.ExtraBold; color: Theme.text }
                        Text { objectName: "count"; text: page.ppl.count || ""; anchors.baseline: h1.baseline; font.family: Theme.body; font.pixelSize: 14; color: Theme.dim }
                    }
                    Text { text: T.t("people.intro"); Layout.fillWidth: true; Layout.maximumWidth: 1050; wrapMode: Text.Wrap; font.family: Theme.body; font.pixelSize: Theme.textSize; lineHeight: 1.2; color: Theme.dim }
                }
                MmButton {
                    objectName: "nextbtn"
                    variant: "primary"; radius: 12
                    topPadding: 11; bottomPadding: 11; leftPadding: 18; rightPadding: 18
                    font.pixelSize: 14
                    text: T.t("people.next")
                    trailingIcon: "arrow-right"
                    onClicked: store.call("go", { page: "editor" })
                }
            }
            GridLayout {
                objectName: "grid"
                visible: (page.ppl.cards || []).length > 0
                Layout.fillWidth: true
                columns: Math.max(1, Math.floor((width + 14) / (212 + 14)))
                columnSpacing: 14; rowSpacing: 14
                Repeater {
                    model: page.ppl.cards || []
                    delegate: PersonCard { required property var modelData; c: modelData; Layout.fillWidth: true; Layout.preferredWidth: 1; Layout.alignment: Qt.AlignTop }
                }
            }
            GridLayout {
                objectName: "briefgrid"
                visible: (page.ppl.brief || []).length > 0
                Layout.fillWidth: true
                columns: Math.max(1, Math.floor((width + 14) / (212 + 14)))
                columnSpacing: 14; rowSpacing: 14
                Repeater {
                    model: page.ppl.brief || []
                    delegate: PersonCard { required property var modelData; c: modelData; Layout.fillWidth: true; Layout.preferredWidth: 1; Layout.alignment: Qt.AlignTop }
                }
            }
            Rectangle {
                objectName: "foot"
                Layout.fillWidth: true
                Layout.bottomMargin: 28
                implicitHeight: footRow.implicitHeight + 28
                radius: 12
                color: Theme.panel
                border.color: Theme.line
                RowLayout {
                    id: footRow
                    anchors.fill: parent
                    anchors.leftMargin: 16; anchors.rightMargin: 16
                    spacing: 8
                    Text { text: page.ppl.others || ""; font.family: Theme.body; font.pixelSize: Theme.small; color: Theme.dim }
                    MmButton {
                        objectName: "showbrief"
                        visible: (page.ppl.brief_button || "") !== ""
                        variant: "link"
                        text: page.ppl.brief_button || ""
                        font.weight: Font.DemiBold
                        onClicked: store.call("people_toggle", { what: "brief" })
                    }
                    Item { Layout.fillWidth: true }
                    Icon { name: "check"; size: 14; color: Theme.faint }
                    Text { text: T.t("people.saved"); font.family: Theme.body; font.pixelSize: Theme.small; color: Theme.faint }
                }
            }
        }
    }

    component PersonCard: Rectangle {
        id: card
        property var c
        objectName: "person-" + c.tid
        implicitHeight: col.implicitHeight
        radius: 14
        color: Theme.panel
        border.color: c.shown ? Theme.good : Theme.line
        clip: true
        ColumnLayout {
            id: col
            width: parent.width
            spacing: 0
            Item {
                Layout.fillWidth: true
                Layout.preferredHeight: width * 176 / 232
                RectCrop { anchors.fill: parent; anchors.margins: 1; pic: card.c.pic; mmPublic: page.proj.public }
                AbstractButton {
                    id: em
                    objectName: "emoji-" + card.c.tid
                    visible: !card.c.shown
                    anchors.right: parent.right; anchors.bottom: parent.bottom
                    anchors.rightMargin: 10; anchors.bottomMargin: 10
                    width: 40; height: 40
                    hoverEnabled: true
                    scale: hovered ? 1.06 : 1
                    onClicked: store.call("people_toggle", { what: "palette", tid: card.c.tid })
                    ToolTip.visible: hovered
                    ToolTip.delay: 600
                    ToolTip.text: T.t("people.pick_emoji")
                    background: Rectangle { radius: 20; color: "#E60D0B17"; border.color: em.hovered ? Theme.accent : "#33FFFFFF" }
                    contentItem: Text { text: card.c.emoji; font.family: "Apple Color Emoji"; font.pixelSize: 24; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                }
            }
            ColumnLayout {
                Layout.fillWidth: true
                Layout.margins: 12
                spacing: 10
                TextField {
                    id: nameField
                    objectName: "name-" + card.c.tid
                    Layout.fillWidth: true
                    text: card.c.name
                    placeholderText: T.t("people.name_placeholder")
                    placeholderTextColor: Theme.faint
                    color: Theme.text
                    font.family: Theme.body; font.pixelSize: Theme.small
                    padding: 8; leftPadding: 10
                    background: Rectangle { radius: 8; color: Theme.deep; border.color: nameField.activeFocus ? Theme.accent : Theme.line }
                    onEditingFinished: if (text.trim() !== card.c.name) store.call("people_say", { tid: card.c.tid, kind: "name", value: text.trim() })
                }
                Flow {
                    objectName: "palette-" + card.c.tid
                    visible: card.c.palette_open && !card.c.shown
                    Layout.fillWidth: true
                    spacing: 2
                    Repeater {
                        model: card.c.palette_open ? page.ppl.palette : []
                        delegate: AbstractButton {
                            id: pe
                            required property string modelData
                            hoverEnabled: true
                            padding: 2; leftPadding: 3; rightPadding: 3
                            onClicked: store.call("people_say", { tid: card.c.tid, kind: "emoji", value: modelData })
                            background: Rectangle { radius: 6; color: pe.hovered ? Theme.raised : "transparent" }
                            contentItem: Text { text: pe.modelData; font.family: "Apple Color Emoji"; font.pixelSize: 22 }
                        }
                    }
                }
                SegTabs {
                    objectName: "mask-" + card.c.tid
                    Layout.fillWidth: true
                    fill: true
                    keys: ["masked", "unmasked"]
                    labels: [T.t("people.masked"), T.t("people.unmasked_button")]
                    current: card.c.shown ? "unmasked" : "masked"
                    accentOn: true
                    goodKey: "unmasked"
                    onPicked: (key) => { if ((key === "unmasked") !== card.c.shown) store.call("people_say", { tid: card.c.tid, kind: "show", value: key === "unmasked" }) }
                }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Text { text: card.c.line; font.family: Theme.body; font.pixelSize: 12; color: Theme.dim; Layout.fillWidth: true; wrapMode: Text.Wrap }
                    MmButton { objectName: "faces-" + card.c.tid; variant: "link"; text: T.t("people.their_faces"); font.pixelSize: 12; onClicked: store.call("people_toggle", { what: "strip", tid: card.c.tid }) }
                    Text { visible: card.c.why; text: "·"; color: Theme.dim; font.pixelSize: 12 }
                    MmButton { objectName: "why-" + card.c.tid; visible: card.c.why; variant: "link"; text: T.t("people.why"); font.pixelSize: 12; onClicked: store.call("go", { page: "decisions", piece: card.c.pieces[0] }) }
                }
                GridLayout {
                    objectName: "strip-" + card.c.tid
                    visible: card.c.strip_open
                    Layout.fillWidth: true
                    columns: 2
                    columnSpacing: 6; rowSpacing: 6
                    Repeater {
                        model: card.c.strip_open ? card.c.strip : []
                        delegate: RectCrop {
                            id: sc
                            required property var modelData
                            Layout.fillWidth: true
                            Layout.preferredHeight: width
                            pic: modelData
                            box: stripHover.hovered
                            mmPublic: page.proj.public
                            HoverHandler { id: stripHover; cursorShape: Qt.PointingHandCursor }
                            TapHandler { onTapped: store.call("edit_at", { t: sc.modelData.t, piece: sc.modelData.piece }) }
                            ToolTip.visible: stripHover.hovered
                            ToolTip.delay: 600
                            ToolTip.text: modelData.tip
                        }
                    }
                }
                Text { visible: card.c.error !== ""; text: card.c.error; color: Theme.bad; font.family: Theme.body; font.pixelSize: 12; Layout.fillWidth: true; wrapMode: Text.Wrap }
                MmButton {
                    visible: card.c.ask !== ""
                    Layout.fillWidth: true
                    text: card.c.ask
                    onClicked: store.call("go", { page: "decisions" })
                    contentItem: Text { text: card.c.ask; wrapMode: Text.Wrap; font.family: Theme.body; font.pixelSize: Theme.small; font.weight: Font.DemiBold; color: Theme.accent }
                    background: Rectangle { radius: Theme.radius; color: "transparent"; border.color: Theme.accent }
                }
            }
        }
    }
}
