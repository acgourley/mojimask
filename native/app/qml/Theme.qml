// The look, from web/theme.css (the current app's tokens): colours, type, radii, spacing. Dark only.
// Change a colour here, not in a page.
pragma Singleton
import QtQuick

QtObject {
    // surfaces
    readonly property color bg: "#0D0B17"
    readonly property color bar: "#131022"
    readonly property color panel: "#15121F"
    readonly property color deep: "#0A0812"
    readonly property color raised: "#221D36"
    readonly property color raised2: "#3A3360"
    readonly property color menu: "#241F3A"
    readonly property color chip: "#1A1628"
    // lines
    readonly property color line: "#2A2540"
    readonly property color line2: "#4A4273"
    readonly property color line3: "#3D3566"
    readonly property color track: "#2F2A48"
    // text
    readonly property color text: "#FFFFFF"
    readonly property color soft: "#C9C4DB"
    readonly property color dim: "#9A93B0"
    readonly property color faint: "#6F6988"
    // accent and states
    readonly property color accent: "#FFD23F"
    readonly property color ink: "#1A1405"
    readonly property color accentSoft: "#26FFD23F"   // --accent-soft #FFD23F26 (Qt: #AARRGGBB)
    readonly property color accentTint: "#1FFFD23F"   // --accent-tint #FFD23F1F
    readonly property color good: "#3DDC84"
    readonly property color warn: "#F5A524"
    readonly property color bad: "#FF6B6B"
    // a box: detected, held, yours
    readonly property color obs: "#3DDC84"
    readonly property color coast: "#FFB020"
    readonly property color manual: "#29B6F6"
    // Decisions' boxes: the green one (the piece asked about), the grey one (the person)
    readonly property color piece: "#3DDC84"
    readonly property color person: "#B8B4C8"
    // the drop card's gradient (app.html .drop)
    readonly property color dropA: "#2A2048"
    readonly property color dropB: "#17132A"

    // type: Nunito (headings, the wordmark) and Inter (everything else) from native/web/fonts; the Mac's own if they
    // don't load
    readonly property FontLoader nunito: FontLoader { source: fontsDir + "Nunito.woff2" }
    readonly property FontLoader inter: FontLoader { source: fontsDir + "Inter.woff2" }
    readonly property string display: nunito.status === FontLoader.Ready ? nunito.name : "SF Pro Rounded"
    readonly property string body: inter.status === FontLoader.Ready ? inter.name : "SF Pro Text"
    readonly property int textSize: 14
    readonly property int small: 13
    readonly property int tiny: 12
    readonly property int h1: 30
    readonly property int h2: 22
    // web pages are laid out at 14 px with line-height 1.5
    readonly property real lineHeight: 1.5

    // shapes and spacing
    readonly property int radius: 10
    readonly property int radiusBig: 14
    readonly property int pad: 12
    readonly property int gap: 8
    readonly property int barHeight: 52
    // room for the window's traffic lights (the bar is the title bar)
    readonly property int trafficLights: 78

    // the wordmark (native/web/brand.js)
    readonly property string brandName: "mojimask"
    readonly property int brandSplit: 4
    readonly property color brandFirst: "#FFFFFF"
    readonly property color brandSecond: "#FFD23F"
}
