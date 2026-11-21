// Decisions (web/decisions.html): the questions the scan left to you, one selected at a time: the list on the left,
// the selected one on the right with its pictures playing, answered with 1 2 3 (or a click). J / K move. Developer
// mode adds the rules' answers, filters by kind and the evidence. What's shown is store.state.decisions (decide.rs
// works it out); what's done is store.call.
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import ".."
import "../components"

Rectangle {
    id: page
    objectName: "page-decisions"
    color: Theme.bg
    readonly property var dec: store.state.decisions
    readonly property var proj: store.state.project
    readonly property var detail: dec.detail
    readonly property real side: Math.max(200, Math.min(page.height - 545, 460))

    // keys 1-3 answer by slot, 0-9 whose face, J / K and the arrows move (decide.rs key)
    Repeater {
        model: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "J", "K", "Up", "Down"]
        delegate: Item {
            required property string modelData
            Shortcut {
                sequence: modelData
                enabled: page.visible && !confirm.opened
                onActivated: store.call("decisions_key", { key: modelData === "J" ? "j" : modelData === "K" ? "k" : modelData })
            }
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.leftMargin: 32; anchors.rightMargin: 32
        anchors.topMargin: 24; anchors.bottomMargin: 28
        spacing: 18

        // the heading: Decisions, the count, what it is; Next
        RowLayout {
            Layout.fillWidth: true
            spacing: 24
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 6
                Row {
                    spacing: 14
                    Text { text: T.t("nav.decisions"); font.family: Theme.display; font.pixelSize: Theme.h1; font.weight: Font.ExtraBold; color: Theme.text }
                    Text { objectName: "count"; text: page.dec.count || ""; anchors.baseline: parent.children[0].baseline; font.family: Theme.body; font.pixelSize: 14; color: Theme.dim }
                }
                Text {
                    text: T.t("decisions.intro")
                    Layout.fillWidth: true
                    Layout.maximumWidth: 1050
                    wrapMode: Text.Wrap
                    font.family: Theme.body; font.pixelSize: Theme.textSize
                    lineHeight: 1.35
                    color: Theme.dim
                }
            }
            MmButton {
                objectName: "nextbtn"
                variant: "primary"
                radius: 12
                topPadding: 11; bottomPadding: 11; leftPadding: 18; rightPadding: 18
                font.pixelSize: 14
                text: T.t("decisions.next")
                trailingIcon: "arrow-right"
                onClicked: store.call("go", { page: "people" })
            }
        }

        // filters: To answer / Answered (developer mode: the rules', all, by kind)
        Flow {
            objectName: "filters"
            Layout.fillWidth: true
            spacing: 8
            Repeater {
                model: (page.dec.filters || []).slice(0, 2)
                delegate: Pill {
                    required property var modelData
                    objectName: "filter-" + modelData.key
                    text: modelData.label + " " + modelData.n
                    on: modelData.on
                    onClicked: store.call("decisions_filter", { filter: modelData.key })
                }
            }
            Item { visible: page.dec.dev; width: 9; height: 32; Rectangle { anchors.centerIn: parent; width: 1; height: 22; color: Theme.line } }
            Item { visible: page.dec.dev; width: devTag1.width; height: 32; DevTag { id: devTag1; anchors.verticalCenter: parent.verticalCenter } }
            Repeater {
                model: (page.dec.filters || []).slice(2)
                delegate: Pill {
                    required property var modelData
                    objectName: "filter-" + modelData.key
                    text: modelData.label + " " + modelData.n
                    on: modelData.on
                    onClicked: store.call("decisions_filter", { filter: modelData.key })
                }
            }
            Repeater {
                model: page.dec.kinds || []
                delegate: Pill {
                    required property var modelData
                    objectName: "kind-" + modelData.key
                    text: modelData.label + " " + modelData.n
                    on: modelData.on
                    onClicked: store.call("decisions_filter", { kind: modelData.key })
                }
            }
        }

        // only one box's decisions (the Editor's R, People's Why): decisions.html #piecebar
        Rectangle {
            objectName: "piecebar"
            visible: (page.dec.piece || "") !== ""
            Layout.fillWidth: true
            implicitHeight: pcol.implicitHeight + 20
            radius: 12
            color: "transparent"
            border.color: Theme.accent
            ColumnLayout {
                id: pcol
                x: 14; y: 10; width: parent.width - 28
                spacing: 4
                Row {
                    spacing: 6
                    Text { textFormat: Text.StyledText; text: page.dec.piece_text || ""; font.family: Theme.body; font.pixelSize: 13; color: Theme.text }
                    Text {
                        objectName: "piece-nearby"
                        visible: (page.dec.nearby_text || "") !== ""
                        text: (page.dec.nearby_text || "") + " ·"; font.family: Theme.body; font.pixelSize: 13; color: Theme.accent
                        MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: store.call("decisions_nearby") }
                    }
                    Text {
                        objectName: "piece-all"
                        text: page.dec.all_text || ""; font.family: Theme.body; font.pixelSize: 13; color: Theme.accent
                        MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: store.call("decisions_piece", { piece: null }) }
                    }
                }
                Text { text: page.dec.split_hint || ""; font.family: Theme.body; font.pixelSize: 12; color: Theme.dim; wrapMode: Text.WordWrap; Layout.fillWidth: true }
            }
        }

        Text {
            objectName: "dec-error"
            visible: (page.dec.error || "") !== ""
            text: page.dec.error || ""
            color: Theme.bad
            font.family: Theme.body; font.pixelSize: Theme.small
            Layout.fillWidth: true
            wrapMode: Text.Wrap
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 20

            // the list
            ListView {
                id: list
                objectName: "list"
                Layout.preferredWidth: Math.max(260, Math.min(380, page.width * 0.3))
                Layout.fillHeight: true
                clip: true
                spacing: 6
                model: page.dec.rows || []
                currentIndex: page.dec.cur
                onCurrentIndexChanged: positionViewAtIndex(currentIndex, ListView.Contain)
                boundsBehavior: Flickable.StopAtBounds
                ScrollBar.vertical: ScrollBar {}
                delegate: AbstractButton {
                    id: qrow
                    required property var modelData
                    required property int index
                    readonly property bool cur: index === page.dec.cur
                    objectName: "qrow-" + index
                    width: list.width - 6
                    height: 74
                    hoverEnabled: true
                    onClicked: store.call("decisions_select", { i: index })
                    background: Rectangle {
                        radius: 12
                        color: qrow.cur ? Theme.raised : Theme.panel
                        border.color: qrow.cur ? Theme.accent : qrow.hovered ? Theme.line2 : Theme.line
                    }
                    contentItem: RowLayout {
                        spacing: 12
                        Crop {
                            Layout.leftMargin: 0
                            Layout.preferredWidth: 52
                            Layout.preferredHeight: 52
                            mmPublic: page.proj.public
                            crop: qrow.modelData.thumb
                        }
                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 3
                            Text { text: qrow.modelData.q; font.family: Theme.body; font.pixelSize: 14; font.weight: Font.DemiBold; color: Theme.text; Layout.fillWidth: true; elide: Text.ElideRight }
                            Text { text: qrow.modelData.why; font.family: Theme.body; font.pixelSize: 12; color: Theme.dim; Layout.fillWidth: true; elide: Text.ElideRight }
                        }
                        Rectangle {
                            visible: qrow.modelData.badge !== ""
                            implicitWidth: badgeText.implicitWidth + 18
                            implicitHeight: badgeText.implicitHeight + 6
                            radius: 8
                            color: qrow.modelData.badge_kind === "user" ? Theme.accentTint : Theme.chip
                            Text {
                                id: badgeText
                                anchors.centerIn: parent
                                text: qrow.modelData.badge
                                font.family: Theme.body; font.pixelSize: 12; font.weight: Font.DemiBold
                                color: qrow.modelData.badge_kind === "user" ? Theme.accent : Theme.dim
                            }
                        }
                    }
                    padding: 10
                }
                footer: Text {
                    width: list.width
                    topPadding: 14; bottomPadding: 8; leftPadding: 2
                    text: page.dec.hint || ""
                    font.family: Theme.body; font.pixelSize: 12
                    color: Theme.faint
                    wrapMode: Text.Wrap
                }
            }

            // the selected question, large
            Rectangle {
                objectName: "detail"
                Layout.fillWidth: true
                Layout.fillHeight: true
                radius: 18
                color: Theme.panel
                border.color: Theme.line

                ColumnLayout {
                    objectName: "emptybox"
                    visible: !page.detail && !!page.dec.empty
                    anchors.centerIn: parent
                    width: Math.min(parent.width - 48, 520)
                    spacing: 14
                    Text { text: page.dec.empty ? page.dec.empty.title : ""; Layout.alignment: Qt.AlignHCenter; font.family: Theme.display; font.pixelSize: Theme.h2; font.weight: Font.ExtraBold; color: Theme.text }
                    Text { text: page.dec.empty ? page.dec.empty.text : ""; Layout.fillWidth: true; horizontalAlignment: Text.AlignHCenter; wrapMode: Text.Wrap; font.family: Theme.body; font.pixelSize: Theme.textSize; color: Theme.dim }
                    MmButton {
                        objectName: "emptynext"
                        visible: page.dec.empty ? page.dec.empty.next : false
                        Layout.alignment: Qt.AlignHCenter
                        variant: "primary"; radius: 12
                        topPadding: 11; bottomPadding: 11; leftPadding: 18; rightPadding: 18
                        font.pixelSize: 14
                        text: T.t("decisions.next")
                        trailingIcon: "arrow-right"
                        onClicked: store.call("go", { page: "people" })
                    }
                }

                ScrollView {
                    id: detailScroll
                    visible: !!page.detail
                    anchors.fill: parent
                    anchors.margins: 24
                    contentWidth: availableWidth
                    clip: true
                    ColumnLayout {
                        width: detailScroll.availableWidth
                        spacing: 20

                        // the pictures: they play while the question is selected; the pointer scrubs them
                        Row {
                            objectName: "pics"
                            spacing: 16
                            Repeater {
                                model: page.detail ? page.detail.pics : []
                                delegate: LoopPic {
                                    required property var modelData
                                    required property int index
                                    objectName: "pic-" + index
                                    pic: modelData
                                    picIndex: index
                                    detailKey: page.detail ? page.detail.key : ""
                                    width: Math.min(page.side, (detailScroll.availableWidth - 16 * ((page.detail ? page.detail.pics.length : 1) - 1)) / Math.max(1, page.detail ? page.detail.pics.length : 1))
                                    mmPublic: page.proj.public
                                }
                            }
                        }

                        ColumnLayout {
                            spacing: 8
                            Layout.fillWidth: true
                            Text {
                                objectName: "q"
                                text: page.detail ? page.detail.title : ""
                                Layout.fillWidth: true
                                wrapMode: Text.Wrap
                                font.family: Theme.display; font.pixelSize: 26; font.weight: Font.ExtraBold
                                lineHeight: 1.15
                                color: Theme.text
                            }
                            Text {
                                objectName: "qnote"
                                text: page.detail ? page.detail.note : ""
                                Layout.fillWidth: true
                                Layout.maximumWidth: 760
                                wrapMode: Text.Wrap
                                font.family: Theme.body; font.pixelSize: 14
                                lineHeight: 1.2
                                color: Theme.dim
                            }
                        }

                        // the answers: one row, keys 1 2 3 left to right
                        RowLayout {
                            objectName: "answers"
                            Layout.fillWidth: true
                            spacing: 12
                            Repeater {
                                model: page.detail ? page.detail.answers : []
                                delegate: AbstractButton {
                                    id: ab
                                    required property var modelData
                                    required property int index
                                    objectName: "answer-" + modelData.key
                                    Layout.fillWidth: true
                                    Layout.preferredWidth: 1
                                    Layout.minimumWidth: 170
                                    hoverEnabled: true
                                    padding: 14; leftPadding: 16; rightPadding: 16
                                    onClicked: store.call("decisions_answer", { id: page.detail.id, answer: modelData.chosen ? null : modelData.value })
                                    ToolTip.visible: hovered && modelData.title !== ""
                                    ToolTip.delay: 600
                                    ToolTip.text: modelData.title
                                    readonly property color fg: modelData.chosen ? Theme.accent : modelData.rule_chosen ? Theme.warn : Theme.text
                                    background: Rectangle {
                                        radius: 12
                                        color: ab.modelData.chosen ? Theme.accentTint : Theme.raised
                                        border.width: 1
                                        border.color: ab.modelData.chosen || ab.hovered ? Theme.accent : ab.modelData.rule_chosen ? Theme.warn : Theme.raised2
                                    }
                                    contentItem: Item {
                                        implicitHeight: abRow.implicitHeight
                                        Row {
                                            id: abRow
                                            anchors.centerIn: parent
                                            spacing: 10
                                            Icon { visible: ab.modelData.chosen; name: "check"; color: Theme.accent; anchors.verticalCenter: parent.verticalCenter }
                                            Text {
                                                text: ab.modelData.label + (ab.modelData.rule_chosen ? "  · " + T.t("decisions.rule") : "")
                                                font.family: Theme.body; font.pixelSize: 15; font.weight: Font.DemiBold
                                                color: ab.fg
                                                anchors.verticalCenter: parent.verticalCenter
                                            }
                                            Kbd { text: ab.modelData.key; tone: ab.modelData.chosen ? "accent" : ab.modelData.rule_chosen ? "warn" : ""; anchors.verticalCenter: parent.verticalCenter }
                                        }
                                    }
                                }
                            }
                        }

                        // whose face: every person offered, as a picture to click
                        GridLayout {
                            objectName: "cands"
                            visible: page.detail ? page.detail.cands.length > 0 : false
                            Layout.fillWidth: true
                            columns: Math.max(1, Math.floor((detailScroll.availableWidth + 8) / 132))
                            columnSpacing: 8; rowSpacing: 8
                            Repeater {
                                model: page.detail ? page.detail.cands : []
                                delegate: AbstractButton {
                                    id: cb
                                    required property var modelData
                                    objectName: "cand-" + modelData.piece
                                    Layout.fillWidth: true
                                    hoverEnabled: true
                                    padding: 6
                                    onClicked: store.call("decisions_answer", { id: page.detail.id, answer: modelData.chosen ? null : modelData.piece })
                                    background: Rectangle { radius: 12; color: cb.modelData.chosen ? Theme.accentTint : Theme.raised; border.color: cb.modelData.chosen ? Theme.accent : cb.hovered ? Theme.line2 : Theme.raised2 }
                                    contentItem: ColumnLayout {
                                        spacing: 4
                                        Crop { Layout.fillWidth: true; Layout.preferredHeight: width; crop: cb.modelData.pic; mmPublic: page.proj.public }
                                        Row {
                                            spacing: 6
                                            Text { text: cb.modelData.name; font.family: Theme.body; font.pixelSize: 13; font.weight: Font.DemiBold; color: Theme.text }
                                            Kbd { visible: cb.modelData.key !== ""; text: cb.modelData.key }
                                        }
                                        Text { text: cb.modelData.why; Layout.fillWidth: true; wrapMode: Text.Wrap; font.family: Theme.body; font.pixelSize: 11; color: Theme.dim }
                                    }
                                }
                            }
                        }

                        Rectangle {
                            visible: page.detail ? page.detail.rule_pick !== "" : false
                            implicitWidth: rp.implicitWidth + 22; implicitHeight: rp.implicitHeight + 10
                            radius: 9; color: "transparent"; border.color: Theme.dim
                            Text { id: rp; anchors.centerIn: parent; text: page.detail ? page.detail.rule_pick : ""; font.family: Theme.body; font.pixelSize: 13; color: Theme.dim }
                        }

                        // why it's asked, in plain words; the numbers behind it in developer mode
                        ColumnLayout {
                            objectName: "why"
                            spacing: 10
                            Layout.fillWidth: true
                            Flow {
                                Layout.fillWidth: true
                                spacing: 8
                                Item { width: whyLabel.width; height: 24; Eyebrow { id: whyLabel; anchors.verticalCenter: parent.verticalCenter; text: T.t("decisions.why_we_ask") } }
                                Chip { text: page.detail ? page.detail.why : "" }
                                Chip { visible: page.detail ? page.detail.agent : false; text: T.t("decisions.agent") }
                            }
                            Flow {
                                visible: page.detail ? page.detail.chips.length > 0 : false
                                Layout.fillWidth: true
                                spacing: 8
                                Item { width: devTag2.width; height: 24; DevTag { id: devTag2; anchors.verticalCenter: parent.verticalCenter } }
                                Repeater {
                                    model: page.detail ? page.detail.chips : []
                                    delegate: Chip { required property var modelData; text: modelData.text; tone: modelData.cls === "" ? "" : "dashed" }
                                }
                            }
                        }
                        Item { Layout.preferredHeight: 4 }
                    }
                }
            }
        }
    }
}
