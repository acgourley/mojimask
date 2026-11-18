// The scan's "Deciding who's who" step, drawn from what it is actually doing (web/decide_viz.js, ported line for line
// to a QML Canvas; the events are scan/viz.py's, from store.call("viz_events")).
//
// The picture is the clip's heads over time: x = the frame number, y = where the head is across the picture (top =
// the left edge). Each piece (a stretch where following a head was certain) is a line; the step then
//   follows every piece past both its ends, frame by frame (green dot: a face found there, blue: the patch still looks
//     the same, grey ring: nothing to go on, held where the camera says) -- a join where two follows land on each other
//   measures every detection's face (white: clear and facing the camera, comparable; dim: a face, not comparable)
//   compares the pieces pairwise: merges (one person), refusals (on screen together: two people), questions
//   settles the people (colours), makes each one's cover (the filled-in frames) and the same-person questions.
// Events arrive as they happen; bursts are spread out so they can be seen, never more than a few seconds behind.
// Under the chart, the faces themselves (crops of the scanned frames: headUrl?t=&x=&y=&s=&px=).
//
// This is drawing, not the app's logic: what to draw comes from the scan's events, as on the web page.
import QtQuick
import QtQuick.Window
import ".."

Rectangle {
    id: dv
    property bool active: false
    property string headUrl: ""
    property bool finishing: false
    color: "#000000"
    clip: true

    readonly property var dwell: ({ pieces: 1100, follow: 90, joins: 1000, faces_start: 400, faces: 35, chains: 0, pairs: 1000, merge: 700,
                                    refuse: 800, ask: 700, people: 1200, cover: 110, same: 1000 })
    readonly property var how: ({ f: "#3ddc84", c: "#3ddc84", l: "#5b8cff", h: "#80868f", x: "#80868f" })
    property var vs: fresh()
    property int fetched: 0
    property bool drainedSent: false
    property real finishingSince: 0
    property string titleText: T.t("viz.start.title")
    property string subText: ""
    property var legend: []

    function fresh() {
        return { n: 1, pieces: {}, at: {}, order: [], parent: {}, chainOf: {}, chainSpan: {}, people: null, dropped: {}, real: {}, realN: 0,
                 faces: {}, joined: {}, anims: [], arcs: [], links: [], scanT: null, scanAt: 0,
                 count: { follows: 0, checks: 0, found: 0, landed: 0, faces: 0, frontal: 0, merges: 0, refused: 0, asks: 0, covers: 0, filled: 0, pairs: 0, same: 0 },
                 stage: "", piecesAt: 0, recolorAt: 0, trailsFadeAt: 0, queue: [], nextAt: 0, trails: [], fills: [], lastFace: 0 }
    }
    function reset() { vs = fresh(); fetched = 0; drainedSent = false; faces.clear(); caption(); tlayer.requestPaint(); flayer.requestPaint() }
    onHeadUrlChanged: reset()
    onActiveChanged: if (active && store.state.scan.viz_n < fetched) reset()

    // ---- geometry
    readonly property var pad: ({ l: 30, r: 12, t: 58, b: 112 })
    function xAt(t) { return pad.l + (width - pad.l - pad.r) * (vs.n > 1 ? t / (vs.n - 1) : 0) }
    function yAt(x) { return pad.t + (height - pad.t - pad.b) * Math.max(-0.02, Math.min(1.02, x)) }
    function lw(s) { return Math.max(1.5, Math.min(4.5, s * 40)) }
    function hue(r) { return (r * 137.508) % 360 }
    function hsla(h, s, l, a) { return Qt.hsla(h / 360, s / 100, l / 100, a) }
    function find(a) { while (vs.parent[a] !== undefined && vs.parent[a] !== a) a = vs.parent[a]; return a }
    function union(a, b) { a = find(a); b = find(b); if (a !== b) vs.parent[b] = a }
    function now() { return Date.now() }
    function head(o) { return headUrl + "?t=" + o[0] + "&x=" + o[1] + "&y=" + o[2] + "&s=" + o[3] + "&px=120" }

    // ---- faces under the chart
    ListModel { id: faces }
    function addCard(c) {
        faces.insert(0, c)
        while (faces.count > 14) faces.remove(faces.count - 1)
    }
    function face(o, label, c, throttle) {
        if (!o) return
        const t = now()
        if (t - vs.lastFace < (throttle === undefined ? 220 : throttle)) return
        vs.lastFace = t
        addCard({ a: head(o), b: "", label: label || "", color: String(c) })
    }
    function pair(a, b, label, c) {
        if (!a || !b) return
        addCard({ a: head(a), b: head(b), label: label, color: String(c) })
        vs.lastFace = now()
    }
    function best(ks) {
        let b = null
        for (const k of ks) for (const o of vs.pieces[k] || []) if (!b || o[3] > b[3]) b = o
        return b
    }

    // ---- events
    function apply(e) {
        const t0 = now()
        switch (e.e) {
        case "pieces":
            vs.n = e.n; vs.pieces = {}; vs.at = {}
            for (const k in e.pieces) {
                const obs = e.pieces[k]
                vs.pieces[k] = obs; vs.at[k] = {}; vs.parent[k] = +k
                for (const o of obs) vs.at[k][o[0]] = o
            }
            vs.order = Object.keys(vs.pieces)
            vs.stage = "follow"; vs.piecesAt = t0; break
        case "follow": {
            vs.count.follows++
            for (const [p, dir] of [[e.fwd, 1], [e.bwd, -1]]) {
                vs.count.checks += p.how.length
                vs.count.found += p.how.split("").filter((c) => c === "f" || c === "c").length
                if (p.met !== null && p.met !== undefined) vs.count.landed++
                vs.anims.push({ k: e.k, p: p, dir: dir, t0: t0 })
            }
            for (const [p, dir] of [[e.fwd, 1], [e.bwd, -1]]) {
                const L = p.x.length, a0 = vs.at[e.k] && vs.at[e.k][p.from], s0 = a0 ? a0[3] : null
                if (!L || s0 === null) continue
                if (p.met !== null && p.met !== undefined) { face([p.from + dir * L, p.x[L - 1], p.y[L - 1], s0], T.t("viz.face.joined"), "#3ddc84"); break }
                const i = p.how.search(/[fc]/)
                if (i >= 0) { face([p.from + dir * (i + 1), p.x[i], p.y[i], s0], T.t("viz.face.found"), "#3ddc84"); break }
                const j = p.how.indexOf("l")
                if (j >= 0) { face([p.from + dir * (j + 1), p.x[j], p.y[j], s0], T.t("viz.face.same_look"), "#5b8cff", 700); break }
            }
            break
        }
        case "joins":
            for (const [a, b] of e.pairs) {
                union(a, b); vs.joined[String(a)] = 1; vs.joined[String(b)] = 1
                const A = vs.pieces[a], B = vs.pieces[b]
                if (A && B) vs.links.push({ a: A[A.length - 1], b: B[0], r: a, t0: t0 })
            }
            vs.recolorAt = t0; break
        case "faces_start": vs.stage = "faces"; vs.trailsFadeAt = t0; break
        case "faces":
            vs.scanT = e.t; vs.scanAt = t0
            for (const [k, lv] of e.d) {
                vs.faces[k + ":" + e.t] = lv
                if (lv) vs.count.faces++
                if (lv === 2) { vs.count.frontal++; face(vs.at[k] && vs.at[k][e.t], T.t("viz.face.clear"), "#ffffff") }
            }
            break
        case "chains":
            vs.scanT = null
            for (const c in e.chains) {
                const ks = e.chains[c]
                let a = Infinity, b = -Infinity, first = null, last = null
                for (const k of ks) {
                    vs.chainOf[k] = +c; union(+c, +k)
                    for (const o of vs.pieces[k] || []) {
                        if (o[0] < a) { a = o[0]; first = o }
                        if (o[0] > b) { b = o[0]; last = o }
                    }
                }
                vs.chainSpan[c] = { first: first, last: last, ks: ks }
            }
            break
        case "pairs": vs.stage = "pairs"; vs.count.pairs = e.compared; break
        case "merge": case "refuse": case "ask": {
            const A = vs.chainSpan[e.a], B = vs.chainSpan[e.b]
            if (!A || !B) break
            const pq = A.first[0] <= B.first[0] ? [A.last, B.first] : [B.last, A.first]
            vs.arcs.push({ kind: e.e, p: pq[0], q: pq[1], t0: t0, r: e.a })
            pair(pq[0], pq[1], { merge: T.t("viz.pair.merge"), refuse: T.t("viz.pair.refuse"), ask: T.t("viz.pair.ask") }[e.e],
                 { merge: "#3ddc84", refuse: "#ff5c5c", ask: "#ffb020" }[e.e])
            if (e.e === "merge") { vs.count.merges++; const a = e.a, b = e.b; later(450, () => { union(a, b); vs.recolorAt = now() }) }
            if (e.e === "refuse") vs.count.refused++
            if (e.e === "ask") vs.count.asks++
            break
        }
        case "people": {
            for (const a of vs.arcs) if (a.kind === "ask") a.until = t0 + 1200
            vs.stage = "people"; vs.people = e.people
            vs.real = {}; for (const r of (e.real || Object.keys(e.people))) vs.real[String(r)] = 1
            vs.realN = Object.keys(vs.real).length
            vs.dropped = {}; for (const d of e.dropped) vs.dropped[String(d)] = 1
            for (const r in e.people) for (const c of e.people[r]) union(+r, c)
            vs.recolorAt = t0
            // the cast: one face per person, biggest view, in their colour
            faces.clear()
            const cast = Object.keys(vs.real).map((r) => [r, best((vs.people[r] || []).reduce((acc, c) => acc.concat(vs.chainSpan[c] ? vs.chainSpan[c].ks : [c]), []))])
                .filter((x) => x[1]).sort((a, b) => a[1][0] - b[1][0])
            cast.forEach(([r, o], i) => later(i * 90, () => faces.append({ a: head(o), b: "", label: "", color: String(hsla(hue(find(+r)), 80, 62, 1)) })))
            break
        }
        case "cover":
            vs.stage = "cover"; vs.count.covers++; vs.count.filled += e.fill.length
            if (e.fill.length) { vs.fills.push({ h: hue(find(e.r)), fill: e.fill }); flayer.requestPaint() }
            break
        case "same": {
            vs.stage = "same"; vs.count.same = e.asked.length
            let shown = 0
            for (const [a, b] of e.asked) {
                const A = vs.chainSpan[a], B = vs.chainSpan[b]
                if (!A || !B) continue
                const pq = A.first[0] <= B.first[0] ? [A.last, B.first] : [B.last, A.first]
                vs.arcs.push({ kind: "same", p: pq[0], q: pq[1], t0: t0, r: a })
                if (shown++ < 3) faces.insert(0, { a: head(best(A.ks) || pq[0]), b: head(best(B.ks) || pq[1]), label: T.t("viz.pair.same"), color: "#ffb020" })
            }
            break
        }
        }
        caption()
    }
    function fmt(v) { return Number(v).toLocaleString(Qt.locale("en_US"), "f", 0) }
    function caption() {
        const c = vs.count, nP = vs.order.length
        const nPeople = vs.people ? Object.keys(vs.people).length : 0, nDropped = Object.keys(vs.dropped).length
        const T2 = {
            follow: [T.t("viz.follow.title"), T.t("viz.follow.sub", { follows: fmt(c.follows), pieces: fmt(nP), checks: fmt(c.checks), found: fmt(c.found), landed: fmt(c.landed) })],
            faces: [T.t("viz.faces.title"), T.t("viz.faces.sub", { faces: fmt(c.faces), frontal: fmt(c.frontal) })],
            pairs: [T.t("viz.pairs.title"), T.t("viz.pairs.sub", { pairs: fmt(c.pairs || 0), merges: c.merges, refused: c.refused, asks: c.asks })],
            people: [T.t("viz.people.title", { n: vs.realN }),
                     [nPeople - vs.realN ? T.t("viz.people.fragments", { n: nPeople - vs.realN }) : "", nDropped ? T.t("viz.people.dropped", { n: nDropped }) : ""].filter(Boolean).join(" · ")],
            cover: [T.t("viz.cover.title"), T.t("viz.cover.sub", { covers: c.covers, filled: fmt(c.filled) })],
            same: [c.same ? T.t("viz.same.title", { n: c.same }) : T.t("viz.same.done"), c.same ? T.t("viz.same.sub") : T.t("viz.same.none")],
        }[vs.stage] || [T.t("viz.start.title"), ""]
        titleText = T2[0]; subText = T2[1]
        legend = vs.stage === "follow" ? [["#3ddc84", T.t("viz.face.found")], ["#5b8cff", T.t("viz.legend.looks_same")], ["ring", T.t("viz.legend.nothing")]]
            : vs.stage === "faces" ? [["#ffffff", T.t("viz.face.clear")], ["#888888", T.t("viz.legend.not_comparable")]]
            : vs.stage === "pairs" ? [["#3ddc84", T.t("viz.pair.merge")], ["#ff5c5c", T.t("viz.pair.refuse")], ["#ffb020", T.t("viz.pair.ask")]]
            : vs.stage === "same" && c.same ? [["#ffb020", T.t("viz.pair.same")]] : []
    }

    // setTimeout for QML: a one-shot timer per call
    Component { id: oneShot; Timer { property var fn; running: true; onTriggered: { fn(); destroy() } } }
    function later(ms, fn) { oneShot.createObject(dv, { interval: Math.max(1, ms), fn: fn }) }

    // ---- colours
    function color(k, t) {
        const r = find(+k)
        const ck = vs.chainOf[k] !== undefined ? vs.chainOf[k] : k
        if (vs.people && vs.dropped[String(ck)]) return "rgba(120,125,135,0.18)"
        if (vs.people && !vs.real[String(r)]) return hsla(hue(r), 25, 55, 0.4)
        const solo = r === +k && !vs.joined[String(k)] && !vs.people
        if (solo && vs.stage === "follow") return "rgba(170,178,190,0.55)"
        const f = Math.min(1, (t - vs.recolorAt) / 500)
        return hsla(hue(r), 55 + 20 * f, 58 + 4 * f, 0.9)
    }

    // ---- the frame loop: play queued events on time (faster when they've piled up), then draw
    FrameAnimation {
        running: dv.active && dv.visible
        onTriggered: dv.tick()
    }
    function tick() {
        const t = now()
        let backlog = vs.queue.reduce((a, e) => a + (dwell[e.e] !== undefined ? dwell[e.e] : 100), 0)
        while (vs.queue.length && t >= vs.nextAt) {
            const e = vs.queue.shift(); apply(e)
            const d = dwell[e.e] !== undefined ? dwell[e.e] : 100
            const speed = Math.max(1, backlog / 2500)
            backlog -= d
            vs.nextAt = Math.max(vs.nextAt, t - 50) + d / speed
        }
        tlayer.opacity = vs.stage === "follow" || !vs.stage ? 1 : Math.max(0.12, 1 - (t - vs.trailsFadeAt) / 1500)
        main.requestPaint()
        // the scan is done: tell the store once everything received has been shown (or after 4 s)
        if (finishing && !drainedSent) {
            if (!finishingSince) finishingSince = t
            if ((!vs.queue.length && !vs.anims.length && fetched >= store.state.scan.viz_n) || t - finishingSince > 4000) {
                drainedSent = true
                store.call("viz_drained")
            }
        }
    }
    Timer {
        interval: 500; repeat: true; triggeredOnStart: true
        running: dv.active
        onTriggered: {
            const r = store.call("viz_events", { since: dv.fetched })
            if (r && r.events && r.events.length) { vs.queue.push(...r.events); dv.fetched += r.events.length }
        }
    }

    // finished follows (trails, faded once the follows are done)
    Canvas {
        id: tlayer
        anchors.fill: parent
        onPaint: {
            const g = getContext("2d")
            g.reset()
            g.globalAlpha = 0.3
            for (const it of vs.trails) for (let i = 0; i < it.p.x.length; i++) {
                g.fillStyle = how[it.p.how[i]] || "#888888"
                g.beginPath(); g.arc(xAt(it.p.from + it.dir * (i + 1)), yAt(it.p.x[i]), 1.4, 0, 7); g.fill()
            }
        }
    }
    // covers (the filled-in frames)
    Canvas {
        id: flayer
        anchors.fill: parent
        onPaint: {
            const g = getContext("2d")
            g.reset()
            g.globalAlpha = 0.22
            const w = Math.max(2, (width - pad.l - pad.r) / vs.n + 0.6)
            for (const it of vs.fills) {
                g.fillStyle = hsla(it.h, 80, 62, 1)
                for (const [t, x] of it.fill) g.fillRect(xAt(t) - w / 2, yAt(x) - 5, w, 10)
            }
        }
    }
    onWidthChanged: { tlayer.requestPaint(); flayer.requestPaint() }
    onHeightChanged: { tlayer.requestPaint(); flayer.requestPaint() }
    Canvas {
        id: main
        anchors.fill: parent
        onPaint: {
            const g = getContext("2d"), t = now(), W = width, H = height
            g.reset()
            // the chart's frame
            g.strokeStyle = "rgba(255,255,255,0.05)"; g.lineWidth = 1
            g.strokeRect(pad.l, pad.t, W - pad.l - pad.r, H - pad.t - pad.b)
            // pieces, sweeping in from the left when they arrive
            const sweep = Math.min(1, (t - vs.piecesAt) / 900) * vs.n
            g.lineCap = "round"; g.lineJoin = "round"
            for (const k of vs.order) {
                const obs = vs.pieces[k]
                if (!obs.length || obs[0][0] > sweep) continue
                g.strokeStyle = color(k, t); g.lineWidth = lw(obs[0][3])
                g.beginPath()
                let prev = null
                for (const o of obs) {
                    if (o[0] > sweep) break
                    if (prev && o[0] - prev[0] <= 2) g.lineTo(xAt(o[0]), yAt(o[1])); else g.moveTo(xAt(o[0]), yAt(o[1]))
                    prev = o
                }
                if (obs.length === 1) g.lineTo(xAt(obs[0][0]) + 0.1, yAt(obs[0][1]))
                g.stroke()
            }
            // joins: the gap between two pieces whose follows met
            for (const l of vs.links) {
                const f = Math.min(1, (t - l.t0) / 400)
                g.strokeStyle = color(l.r, t); g.lineWidth = 2
                if (g.setLineDash) g.setLineDash([3, 3])
                g.beginPath(); g.moveTo(xAt(l.a[0]), yAt(l.a[1]))
                g.lineTo(xAt(l.a[0] + (l.b[0] - l.a[0]) * f), yAt(l.a[1] + (l.b[1] - l.a[1]) * f)); g.stroke()
                if (g.setLineDash) g.setLineDash([])
            }
            // face measurements
            for (const key in vs.faces) {
                const lv = vs.faces[key]
                if (!lv) continue
                const parts = key.split(":"), o = vs.at[parts[0]] && vs.at[parts[0]][parts[1]]
                if (!o) continue
                if (lv === 1 && vs.stage !== "faces") continue  // once measured, only the comparable faces stay, faintly
                g.fillStyle = lv === 2 ? (vs.stage === "faces" ? "#ffffff" : "rgba(255,255,255,0.35)") : "rgba(200,200,200,0.45)"
                g.beginPath(); g.arc(xAt(+parts[1]), yAt(o[1]), lv === 2 ? 2.3 : 1.4, 0, 7); g.fill()
            }
            if (vs.scanT !== null) {
                const x = xAt(vs.scanT), a = Math.max(0, 1 - (t - vs.scanAt) / 1500)
                const gr = g.createLinearGradient(x - 26, 0, x, 0)
                gr.addColorStop(0, "rgba(255,255,255,0)"); gr.addColorStop(1, "rgba(255,255,255," + (0.16 * a) + ")")
                g.fillStyle = gr; g.fillRect(x - 26, pad.t, 26, H - pad.t - pad.b)
                g.strokeStyle = "rgba(255,255,255," + (0.6 * a) + ")"; g.lineWidth = 1
                g.beginPath(); g.moveTo(x, pad.t); g.lineTo(x, H - pad.b); g.stroke()
            }
            // follows in progress: one frame per 45 ms, the head's dot leading
            let baked = false
            vs.anims = vs.anims.filter((a) => {
                const steps = a.p.x.length, shown = Math.min(steps, Math.floor((t - a.t0) / 45))
                for (let i = 0; i < shown; i++) {
                    const c = how[a.p.how[i]] || "#888888", tt = a.p.from + a.dir * (i + 1)
                    if (a.p.how[i] === "h" || a.p.how[i] === "x") {
                        g.strokeStyle = c; g.lineWidth = 1; g.beginPath(); g.arc(xAt(tt), yAt(a.p.x[i]), 2.2, 0, 7); g.stroke()
                    } else { g.fillStyle = c; g.beginPath(); g.arc(xAt(tt), yAt(a.p.x[i]), i === shown - 1 ? 3.4 : 2.2, 0, 7); g.fill() }
                }
                if (shown >= steps) {
                    const end = t - a.t0 - steps * 45
                    if (a.p.met !== null && a.p.met !== undefined && steps && end < 700) {  // landed on another piece: a ring at the landing
                        const i = steps - 1, tt = a.p.from + a.dir * steps
                        g.strokeStyle = "rgba(61,220,132," + (1 - end / 700) + ")"; g.lineWidth = 2
                        g.beginPath(); g.arc(xAt(tt), yAt(a.p.x[i]), 5 + end / 40, 0, 7); g.stroke()
                    }
                    if (end >= 700 || !steps) { if (steps) { vs.trails.push({ p: a.p, dir: a.dir }); baked = true } return false }
                }
                return true
            })
            if (baked) tlayer.requestPaint()
            // comparisons: arcs from where one piece ends to where the other starts
            vs.arcs = vs.arcs.filter((a) => {
                const life = a.kind === "merge" ? 1400 : a.kind === "refuse" ? 1600 : a.kind === "ask" && a.until ? a.until - a.t0 : Infinity
                const age = t - a.t0
                if (age > life) return false
                const f = Math.min(1, age / 450), fade = life === Infinity ? 1 : Math.min(1, (life - age) / 400)
                const c = { merge: "61,220,132", refuse: "255,92,92", ask: "255,176,32", same: "255,176,32" }[a.kind]
                const x1 = xAt(a.p[0]), y1 = yAt(a.p[1]), x2 = xAt(a.q[0]), y2 = yAt(a.q[1])
                const mx = (x1 + x2) / 2, my = Math.max(pad.t + 8, Math.min(y1, y2) - 18 - Math.abs(x2 - x1) * 0.15)
                g.strokeStyle = "rgba(" + c + "," + (0.95 * fade) + ")"; g.lineWidth = 2
                if ((a.kind === "ask" || a.kind === "same") && g.setLineDash) g.setLineDash([5, 4])
                g.beginPath(); g.moveTo(x1, y1)
                const N = 24
                for (let i = 1; i <= N * f; i++) {
                    const s = i / N, x = (1 - s) * (1 - s) * x1 + 2 * s * (1 - s) * mx + s * s * x2, y = (1 - s) * (1 - s) * y1 + 2 * s * (1 - s) * my + s * s * y2
                    g.lineTo(x, y)
                }
                g.stroke(); if (g.setLineDash) g.setLineDash([])
                if (f >= 1 && a.kind !== "merge") {
                    const lx = 0.25 * x1 + 0.5 * mx + 0.25 * x2, ly = 0.25 * y1 + 0.5 * my + 0.25 * y2
                    g.fillStyle = "rgba(" + c + "," + fade + ")"; g.font = "bold 13px sans-serif"; g.textAlign = "center"
                    g.fillText(a.kind === "refuse" ? "×" : "?", lx, ly - 4)
                }
                return true
            })
        }
    }

    // the caption
    Column {
        x: 14; y: 10
        width: parent.width - 28
        spacing: 0
        Text { objectName: "viz-title"; text: dv.titleText; width: parent.width; elide: Text.ElideRight; font.family: Theme.body; font.pixelSize: 14; font.weight: Font.DemiBold; color: "#ececec" }
        Text { objectName: "viz-sub"; text: dv.subText; width: parent.width; wrapMode: Text.Wrap; height: Math.max(18, implicitHeight); font.family: Theme.body; font.pixelSize: 12; font.features: { "tnum": 1 }; color: Theme.dim }
    }
    // the faces
    ListView {
        id: facesView
        x: 30
        y: parent.height - 24 - 84
        width: parent.width - 42
        height: 84
        orientation: ListView.Horizontal
        spacing: 8
        interactive: false
        clip: true
        model: faces
        add: Transition { NumberAnimation { property: "opacity"; from: 0; to: 1; duration: 350; easing.type: Easing.OutCubic } NumberAnimation { property: "scale"; from: 0.9; to: 1; duration: 350; easing.type: Easing.OutCubic } }
        displaced: Transition { NumberAnimation { properties: "x"; duration: 350; easing.type: Easing.OutCubic } }
        delegate: Column {
            id: fcard
            required property string a
            required property string b
            required property string label
            required property string color
            spacing: 2
            height: 84
            Item { width: 1; height: 84 - row.height - (lab.visible ? lab.height + 2 : 0) }
            Row {
                id: row
                spacing: 2
                anchors.horizontalCenter: parent.horizontalCenter
                Repeater {
                    model: fcard.b ? [fcard.a, fcard.b] : [fcard.a]
                    delegate: Rectangle {
                        required property string modelData
                        width: fcard.b ? 48 : 60; height: width; radius: 8
                        color: "#111111"
                        border.width: 2
                        border.color: fcard.color || "#444444"
                        clip: true
                        Image { anchors.fill: parent; anchors.margins: 2; source: modelData; fillMode: Image.PreserveAspectCrop; sourceSize: Qt.size(width * Screen.devicePixelRatio, height * Screen.devicePixelRatio); asynchronous: true; smooth: true }
                    }
                }
            }
            Text { id: lab; visible: label !== ""; text: label; anchors.horizontalCenter: parent.horizontalCenter; font.family: Theme.body; font.pixelSize: 10; color: fcard.color || Theme.dim }
        }
    }
    // the legend and the axes
    Row {
        anchors.right: parent.right; anchors.rightMargin: 12
        anchors.bottom: parent.bottom; anchors.bottomMargin: 5
        spacing: 6
        Repeater {
            model: dv.legend
            delegate: Row {
                required property var modelData
                spacing: 6
                leftPadding: 6
                Rectangle {
                    width: 7; height: 7; radius: 3.5; anchors.verticalCenter: parent.verticalCenter
                    color: modelData[0] === "ring" ? "transparent" : modelData[0]
                    border.width: modelData[0] === "ring" ? 1 : 0; border.color: "#80868f"
                }
                Text { text: modelData[1]; font.family: Theme.body; font.pixelSize: 11; color: Theme.dim }
            }
        }
    }
    Text { x: 30; anchors.bottom: parent.bottom; anchors.bottomMargin: 5; text: T.t("viz.axis_x"); font.family: Theme.body; font.pixelSize: 11; color: Theme.faint }
    Text {
        text: T.t("viz.axis_y")
        font.family: Theme.body; font.pixelSize: 11; color: Theme.faint
        rotation: -90
        x: 4 + 6 - width / 2
        y: dv.pad.t + (dv.height - dv.pad.t - dv.pad.b) / 2 - height / 2
    }
}
