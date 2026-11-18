// The scan (web/app.html #s-scan): the live preview (then "Deciding who's who", drawn from what the step is doing),
// how far it is, the steps, the numbers, and Cancel. A project scanned before only shows "Opening…" while its server
// starts. Everything shown is store.state.scan.
import QtQuick
import QtQuick.Window
import QtQuick.Controls.Basic
import QtQuick.Layouts
import QtQuick.Effects
import ".."
import "../components"

Rectangle {
    id: page
    objectName: "page-scan"
    color: Theme.bg
    readonly property var sc: store.state.scan

    // opening a project scanned before: nothing to show but a moment's wait
    ColumnLayout {
        objectName: "opening"
        visible: page.sc.opening
        anchors.centerIn: parent
        spacing: 10
        Spinner { Layout.alignment: Qt.AlignHCenter }
        Text {
            text: page.sc.opening_name
            Layout.alignment: Qt.AlignHCenter
            Layout.topMargin: 8
            font.family: Theme.display; font.pixelSize: Theme.h2; font.weight: Font.ExtraBold
            color: Theme.text
        }
        Text { text: T.t("scan.opening"); Layout.alignment: Qt.AlignHCenter; font.family: Theme.body; font.pixelSize: Theme.small; color: Theme.dim }
    }

    RowLayout {
        id: scanRow
        objectName: "scanview"
        visible: !page.sc.opening
        width: Math.min(1400, page.width - 64)
        anchors.centerIn: parent
        spacing: 32

        // the picture at the video's own shape, the timeline as wide as it
        Item {
            Layout.fillWidth: true
            Layout.preferredWidth: 62
            Layout.preferredHeight: stagecol.height
            ColumnLayout {
                id: stagecol
                readonly property real ar: page.sc.aspect || 1.6
                width: Math.min(parent.width, Math.min(page.height * 0.68, 760) * ar)
                anchors.horizontalCenter: parent.horizontalCenter
                spacing: 14
                Picture {
                    id: stage
                    objectName: "stage"
                    mmPublic: page.sc.public
                    Layout.fillWidth: true
                    Layout.preferredHeight: width / stagecol.ar
                    Item {
                        id: stageContent
                        anchors.fill: parent
                        clip: true
                        Rectangle { anchors.fill: parent; color: "#000000" }
                        ColumnLayout {
                            visible: !preview.shown && !page.sc.viz
                            anchors.centerIn: parent
                            width: parent.width - 56
                            spacing: 14
                            Spinner { Layout.alignment: Qt.AlignHCenter }
                            Text {
                                objectName: "previewnote"
                                text: page.sc.note
                                Layout.fillWidth: true
                                horizontalAlignment: Text.AlignHCenter
                                wrapMode: Text.Wrap
                                font.family: Theme.body; font.pixelSize: Theme.small
                                color: Theme.dim
                            }
                        }
                        Image {
                            id: preview
                            objectName: "preview"
                            property bool shown: status === Image.Ready && source != ""
                            anchors.fill: parent
                            visible: !page.sc.viz && shown
                            fillMode: Image.PreserveAspectFit
                            sourceSize: Qt.size(width * Screen.devicePixelRatio, height * Screen.devicePixelRatio)
                            source: page.sc.preview ? page.sc.preview + "?n=" + page.sc.preview_n : ""
                            cache: false
                            asynchronous: true
                            retainWhileLoading: true
                            smooth: true
                        }
                        DecideViz {
                            objectName: "decide-viz"
                            anchors.fill: parent
                            visible: page.sc.viz
                            active: page.sc.viz && page.visible
                            headUrl: page.sc.head_url
                            finishing: page.sc.finishing
                        }
                        Rectangle {
                            objectName: "scanbadge"
                            visible: page.sc.badge !== "" && preview.shown && !page.sc.viz
                            x: 14; y: 14
                            width: badgeRow.implicitWidth + 20; height: badgeRow.implicitHeight + 10
                            radius: 8
                            color: "#CC07060D"
                            Row {
                                id: badgeRow
                                anchors.centerIn: parent
                                spacing: 8
                                Rectangle {
                                    width: 7; height: 7; radius: 3.5; color: Theme.accent
                                    anchors.verticalCenter: parent.verticalCenter
                                    SequentialAnimation on opacity { loops: Animation.Infinite; NumberAnimation { to: 0.35; duration: 700; easing.type: Easing.InOutSine } NumberAnimation { to: 1; duration: 700; easing.type: Easing.InOutSine } }
                                }
                                Text { text: page.sc.badge; font.family: Theme.body; font.pixelSize: 12; font.weight: Font.DemiBold; font.features: { "tnum": 1 }; color: Theme.text }
                            }
                        }
                    }
                    RoundCorners { anchors.fill: parent; radius: 14 }
                }
                Rectangle {
                    objectName: "scantl"
                    visible: !page.sc.photo
                    Layout.fillWidth: true
                    height: 6; radius: 3
                    color: Theme.raised
                    Rectangle {
                        width: parent.width * page.sc.timeline; height: parent.height; radius: 3; color: Theme.accent
                        Behavior on width { NumberAnimation { duration: 600 } }
                    }
                }
                RowLayout {
                    visible: !page.sc.photo
                    Layout.fillWidth: true
                    Text { text: page.sc.tl_left; font.family: Theme.body; font.pixelSize: 12; font.features: { "tnum": 1 }; color: Theme.faint }
                    Item { Layout.fillWidth: true }
                    Text { text: page.sc.tl_right; font.family: Theme.body; font.pixelSize: 12; font.features: { "tnum": 1 }; color: Theme.faint }
                }
            }
        }

        ColumnLayout {
            Layout.preferredWidth: 38
            Layout.fillWidth: true
            Layout.minimumWidth: 300
            Layout.maximumWidth: 460
            Layout.alignment: Qt.AlignVCenter
            spacing: 20
            Text {
                objectName: "scanfile"
                text: page.sc.file
                Layout.fillWidth: true
                elide: Text.ElideMiddle
                font.family: Theme.body; font.pixelSize: 11; font.weight: Font.DemiBold; font.letterSpacing: 1.5; font.capitalization: Font.AllUppercase
                color: Theme.faint
            }
            Text {
                objectName: "scanhead"
                text: page.sc.headline
                Layout.fillWidth: true
                Layout.topMargin: -8
                wrapMode: Text.Wrap
                font.family: Theme.display; font.pixelSize: 34; font.weight: Font.ExtraBold
                lineHeight: 1.1
                color: Theme.text
            }
            Row {
                spacing: 10
                Text {
                    id: pctText
                    objectName: "scanpct"
                    text: page.sc.pct
                    font.family: Theme.display; font.pixelSize: 72; font.weight: Font.Black; font.features: { "tnum": 1 }
                    color: Theme.accent
                }
                Text {
                    objectName: "scanleft"
                    text: page.sc.left
                    anchors.baseline: pctText.baseline
                    font.family: Theme.body; font.pixelSize: 14
                    color: Theme.dim
                }
            }
            ColumnLayout {
                spacing: 8
                Layout.fillWidth: true
                RowLayout {
                    objectName: "steps"
                    Layout.fillWidth: true
                    spacing: 4
                    // a count, not the array: the store sends a new array with every update, and a Repeater over it
                    // would rebuild the bars each time, so each one animated up from 0 again
                    Repeater {
                        model: page.sc.steps.length
                        delegate: Rectangle {
                            required property int index
                            Layout.fillWidth: true
                            height: 4; radius: 2
                            color: Theme.line
                            clip: true
                            Rectangle {
                                width: parent.width * (page.sc.steps[index] || 0); height: 4; radius: 2; color: Theme.accent
                                Behavior on width { NumberAnimation { duration: 400 } }
                            }
                        }
                    }
                }
                Text { objectName: "stepcap"; text: page.sc.stepcap; font.family: Theme.body; font.pixelSize: 12; color: Theme.dim; Layout.fillWidth: true; wrapMode: Text.Wrap }
            }
            Rectangle {
                objectName: "stats"
                visible: page.sc.stats.length > 0
                Layout.fillWidth: true
                implicitHeight: statsCol.implicitHeight + 12
                radius: 14
                color: Theme.panel
                border.color: Theme.line
                Column {
                    id: statsCol
                    x: 16
                    y: 6
                    width: parent.width - 32
                    Repeater {
                        model: page.sc.stats
                        delegate: Item {
                            required property var modelData
                            required property int index
                            width: statsCol.width
                            height: 46
                            Rectangle { visible: index > 0; width: parent.width; height: 1; color: Theme.line }
                            Text { anchors.verticalCenter: parent.verticalCenter; text: modelData[0]; font.family: Theme.body; font.pixelSize: 14; color: Theme.dim }
                            Text {
                                anchors.verticalCenter: parent.verticalCenter; anchors.right: parent.right
                                text: modelData[1]
                                font.family: Theme.body; font.pixelSize: 14; font.weight: Font.DemiBold; font.features: { "tnum": 1 }
                                color: Theme.text
                            }
                        }
                    }
                }
            }
            RowLayout {
                visible: page.sc.error !== ""
                Layout.fillWidth: true
                spacing: 10
                Text {
                    objectName: "scanerr"
                    text: page.sc.error
                    Layout.fillWidth: true
                    wrapMode: Text.Wrap
                    font.family: Theme.body; font.pixelSize: Theme.small
                    color: Theme.warn
                }
            }
            Row {
                spacing: 8
                Icon { name: "lock"; size: 14; color: Theme.faint; anchors.verticalCenter: parent.verticalCenter }
                Text {
                    text: T.t("scan.privacy_html").replace(/<[^>]+>/g, "").replace("computer", "Mac")
                    font.family: Theme.body; font.pixelSize: Theme.small
                    color: Theme.faint
                }
            }
            MmButton {
                objectName: "cancelscan"
                visible: page.sc.can_cancel
                big: true
                text: T.t("scan.cancel")
                onClicked: win.ask(T.t("scan.cancel.ask"), T.t("scan.cancel.ok"), true, T.t("scan.cancel.keep"), () => store.call("cancel_scan"))
            }
        }
    }
}
