// The words, icons and name the pages show: native/web/strings.js, icons.js and brand.js, as the store read them
// (store.call("ui_text")). T.t("scan.left", {eta: "3 min"}) is strings.js's t(): {placeholders} from vars ({app} is the
// name), a plural ({one, other}) picked by vars.n. T.icon("play", color, size) is icons.js's Icon() as an SVG data URL.
pragma Singleton
import QtQuick

QtObject {
    readonly property var tables: store.call("ui_text") || ({ strings: {}, icons: {}, brand: {} })
    readonly property string name: tables.brand.name || "mojimask"

    function t(key, vars) {
        let s = tables.strings[key]
        if (s === undefined) return key
        if (typeof s === "object") s = (vars && Number(vars.n) === 1 ? s.one : s.other) || s.other
        const v = Object.assign({ app: name, app_file: name.toLowerCase() }, vars || {})
        return s.replace(/\{(\w+)\}/g, (m, k) => (k in v ? String(v[k]) : m))
    }

    // strings that carry markup (keys ending _html) as Qt's rich text
    function html(key, vars) { return t(key, vars) }

    function icon(name, color) {
        const body = tables.icons[name]
        if (!body) return ""
        const c = String(color).length === 9 ? "#" + String(color).slice(3) : String(color)  // #aarrggbb -> #rrggbb
        const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="' + c +
            '" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' + body + '</svg>'
        return "data:image/svg+xml;utf8," + encodeURIComponent(svg)
    }
}
