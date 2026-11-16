// The scan's "Deciding who's who" step, drawn from what it is actually doing (events from scan/viz.py).
//
// The picture is the clip's heads over time: x = the frame number, y = where the head is across the picture (top =
// the left edge). Each piece (a stretch where following a head was certain) is a line; the step then
//   follows every piece past both its ends, frame by frame (green dot: a face found there, blue: the patch still looks
//     the same, grey ring: nothing to go on, held where the camera says) -- a join where two follows land on each other
//   measures every detection's face (white: clear and facing the camera, comparable; dim: a face, not comparable)
//   compares the pieces pairwise: merges (one person), refusals (on screen together: two people), questions
//   settles the people (colours), makes each one's cover (the filled-in frames) and the same-person questions.
// Events arrive as they happen; bursts (the comparisons take a fraction of a second) are spread out so they can be
// seen, never more than a few seconds behind.
//
// Under the chart, the faces themselves (crops of the scanned frames, opts.head([t, x, y, s]) -> an image URL): where a
// follow found a face or landed, each clear face as it's measured, the two ends each comparison looked at, and at the
// end one face per person.
//
//   const v = DecideViz.mount(element, { head }); v.push(events); await v.drained(3000); v.stop();
window.DecideViz = (() => {
  const DWELL = { pieces: 1100, follow: 90, joins: 1000, faces_start: 400, faces: 35, chains: 0, pairs: 1000, merge: 700,
                  refuse: 800, ask: 700, people: 1200, cover: 110, same: 1000 };
  const HOW = { f: "#3ddc84", c: "#3ddc84", l: "#5b8cff", h: "#80868f", x: "#80868f" };
  const hue = (r) => (r * 137.508) % 360;

  function mount(host, opts = {}) {
    host.innerHTML = `<div class="dv"><canvas></canvas>
      <div class="dv-cap"><div class="dv-title"></div><div class="dv-sub"></div></div>
      <div class="dv-faces"></div><div class="dv-legend"></div><div class="dv-axis-x">${t("viz.axis_x")}</div><div class="dv-axis-y">${t("viz.axis_y")}</div></div>`;
    const root = host.firstChild, cv = root.querySelector("canvas"), ctx = cv.getContext("2d");
    const facesEl = root.querySelector(".dv-faces");
    const titleEl = root.querySelector(".dv-title"), subEl = root.querySelector(".dv-sub"), legendEl = root.querySelector(".dv-legend");
    let W = 0, H = 0, dpr = 1, raf = 0, stopped = false;
    // baked: finished follows (trails, faded once the follows are done) and covers (fills)
    const layer = document.createElement("canvas"), lctx = layer.getContext("2d");
    const tlayer = document.createElement("canvas"), tctx = tlayer.getContext("2d");

    // ---- state
    let n = 1, pieces = {}, at = {}, order = [];  // piece -> [[t, x, y, s]]; piece -> {t: [x, s]}
    const parent = {};
    const find = (a) => { while (parent[a] !== undefined && parent[a] !== a) a = parent[a]; return a; };
    const union = (a, b) => { a = find(a); b = find(b); if (a !== b) parent[b] = a; };
    let chainOf = {}, chainSpan = {}, people = null, dropped = new Set(), real = new Set();
    const faces = {};  // "k:t" -> 1 (a face) | 2 (clear, frontal)
    const joined = new Set();  // pieces a join touched
    let anims = [], arcs = [], links = [], scanT = null, scanAt = 0;
    const count = { follows: 0, checks: 0, found: 0, landed: 0, faces: 0, frontal: 0, merges: 0, refused: 0, asks: 0, covers: 0, filled: 0 };
    let stage = "", piecesAt = 0, recolorAt = 0, trailsFadeAt = 0;
    const queue = []; let nextAt = 0;

    // ---- geometry
    const pad = { l: 30, r: 12, t: 58, b: opts.head ? 112 : 26 };
    const X = (t) => pad.l + (W - pad.l - pad.r) * (n > 1 ? t / (n - 1) : 0);
    const Y = (x) => pad.t + (H - pad.t - pad.b) * Math.max(-0.02, Math.min(1.02, x));
    const width = (s) => Math.max(1.5, Math.min(4.5, s * 40));
    function resize() {
      const r = root.getBoundingClientRect();
      dpr = window.devicePixelRatio || 1; W = r.width; H = r.height;
      for (const c of [cv, layer, tlayer]) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
      for (const c of [ctx, lctx, tctx]) c.setTransform(dpr, 0, 0, dpr, 0, 0);
      rebake();
    }
    const baked = [];  // what's on the layer, to redraw it after a resize
    function bake(item) { baked.push(item); drawBaked(item); }
    function rebake() { lctx.clearRect(0, 0, W, H); tctx.clearRect(0, 0, W, H); baked.forEach(drawBaked); }
    function drawBaked(it) {
      if (it.kind === "trail") {
        for (let i = 0; i < it.p.x.length; i++) {
          const c = HOW[it.p.how[i]] || "#888";
          tctx.globalAlpha = 0.3; tctx.fillStyle = c;
          tctx.beginPath(); tctx.arc(X(it.p.from + it.dir * (i + 1)), Y(it.p.x[i]), 1.4, 0, 7); tctx.fill();
        }
        tctx.globalAlpha = 1;
      } else if (it.kind === "fill") {
        lctx.globalAlpha = 0.22; lctx.fillStyle = `hsl(${it.h},80%,62%)`;
        const w = Math.max(2, (W - pad.l - pad.r) / n + 0.6);
        for (const [t, x] of it.fill) lctx.fillRect(X(t) - w / 2, Y(x) - 5, w, 10);
      }
      lctx.globalAlpha = 1;
    }

    // ---- faces
    let lastFace = 0;
    const MAXF = 14;
    function img(o, hueStr) {
      const im = new Image(); im.src = opts.head(o); im.alt = "";
      if (hueStr) im.style.borderColor = hueStr;
      return im;
    }
    function addCard(el) {
      el.classList.add("dv-in"); facesEl.prepend(el);
      while (facesEl.children.length > MAXF) facesEl.lastChild.remove();
    }
    function face(o, label, c, throttle = 220) {  // one head; skipped when one was shown < throttle ms ago
      if (!opts.head || !o) return;
      const now = performance.now();
      if (now - lastFace < throttle) return;
      lastFace = now;
      const d = document.createElement("div"); d.className = "dv-card";
      d.append(img(o, c));
      if (label) { const sp = document.createElement("span"); sp.textContent = label; sp.style.color = c; d.append(sp); }
      addCard(d);
    }
    function pair(a, b, label, c) {  // the two ends a comparison looked at
      if (!opts.head || !a || !b) return;
      const d = document.createElement("div"); d.className = "dv-card dv-pair";
      const row = document.createElement("div"); row.append(img(a, c), img(b, c));
      const sp = document.createElement("span"); sp.textContent = label; sp.style.color = c;
      d.append(row, sp); addCard(d); lastFace = performance.now();
    }
    const best = (ks) => {  // the biggest view among these pieces
      let b = null;
      for (const k of ks) for (const o of pieces[k] || []) if (!b || o[3] > b[3]) b = o;
      return b;
    };

    // ---- events
    function apply(e) {
      const now = performance.now();
      switch (e.e) {
        case "pieces":
          n = e.n; pieces = {}; at = {};
          for (const [k, obs] of Object.entries(e.pieces)) {
            pieces[k] = obs; at[k] = {}; parent[k] = +k;
            for (const o of obs) at[k][o[0]] = o;
          }
          order = Object.keys(pieces);
          stage = "follow"; piecesAt = now; break;
        case "follow": {
          count.follows++;
          for (const [p, dir] of [[e.fwd, 1], [e.bwd, -1]]) {
            count.checks += p.how.length;
            count.found += [...p.how].filter((c) => c === "f" || c === "c").length;
            if (p.met != null) count.landed++;
            anims.push({ k: e.k, p, dir, t0: now });
          }
          for (const [p, dir] of [[e.fwd, 1], [e.bwd, -1]]) {
            const L = p.x.length, s0 = at[e.k]?.[p.from]?.[3];
            if (!L || s0 == null) continue;
            if (p.met != null) { face([p.from + dir * L, p.x[L - 1], p.y[L - 1], s0], t("viz.face.joined"), "#3ddc84"); break; }
            const i = p.how.search(/[fc]/);
            if (i >= 0) { face([p.from + dir * (i + 1), p.x[i], p.y[i], s0], t("viz.face.found"), "#3ddc84"); break; }
            const j = p.how.indexOf("l");
            if (j >= 0) { face([p.from + dir * (j + 1), p.x[j], p.y[j], s0], t("viz.face.same_look"), "#5b8cff", 700); break; }
          }
          break;
        }
        case "joins":
          for (const [a, b] of e.pairs) {
            union(a, b); joined.add(String(a)); joined.add(String(b));
            const A = pieces[a], B = pieces[b];
            if (A && B) links.push({ a: A[A.length - 1], b: B[0], r: a, t0: now });
          }
          recolorAt = now; break;
        case "faces_start": stage = "faces"; trailsFadeAt = now; break;
        case "faces":
          scanT = e.t; scanAt = now;
          for (const [k, lv] of e.d) {
            faces[k + ":" + e.t] = lv;
            if (lv) count.faces++;
            if (lv === 2) { count.frontal++; face(at[k]?.[e.t], t("viz.face.clear"), "#fff"); }
          }
          break;
        case "chains":
          scanT = null;
          for (const [c, ks] of Object.entries(e.chains)) {
            let a = Infinity, b = -Infinity, first = null, last = null;
            for (const k of ks) {
              chainOf[k] = +c; union(+c, +k);
              for (const o of pieces[k] || []) {
                if (o[0] < a) { a = o[0]; first = o; }
                if (o[0] > b) { b = o[0]; last = o; }
              }
            }
            chainSpan[c] = { first, last, ks };
          }
          break;
        case "pairs": stage = "pairs"; count.pairs = e.compared; break;
        case "merge": case "refuse": case "ask": {
          const A = chainSpan[e.a], B = chainSpan[e.b];
          if (!A || !B) break;
          const [p, q] = A.first[0] <= B.first[0] ? [A.last, B.first] : [B.last, A.first];
          arcs.push({ kind: e.e, p, q, t0: now, r: e.a, why: e.why });
          pair(p, q, { merge: t("viz.pair.merge"), refuse: t("viz.pair.refuse"), ask: t("viz.pair.ask") }[e.e],
               { merge: "#3ddc84", refuse: "#ff5c5c", ask: "#ffb020" }[e.e]);
          if (e.e === "merge") { count.merges++; setTimeout(() => { union(e.a, e.b); recolorAt = performance.now(); }, 450); }
          if (e.e === "refuse") count.refused++;
          if (e.e === "ask") count.asks++;
          break;
        }
        case "people":
          for (const a of arcs) if (a.kind === "ask") a.until = now + 1200;
          stage = "people"; people = e.people; real = new Set((e.real || Object.keys(e.people)).map(String)); dropped = new Set(e.dropped.map(String));
          for (const [r, cs] of Object.entries(e.people)) for (const c of cs) union(+r, c);
          recolorAt = now;
          if (opts.head) {  // the cast: one face per person, biggest view, in their colour
            facesEl.innerHTML = "";
            const cast = [...real].map((r) => [r, best((people[r] || []).flatMap((c) => chainSpan[c]?.ks || [c]))])
              .filter((x) => x[1]).sort((a, b) => a[1][0] - b[1][0]);
            cast.forEach(([r, o], i) => setTimeout(() => {
              const d = document.createElement("div"); d.className = "dv-card dv-in";
              d.append(img(o, `hsl(${hue(find(+r))},80%,62%)`)); facesEl.append(d);
            }, i * 90));
          }
          break;
        case "cover":
          stage = "cover"; count.covers++; count.filled += e.fill.length;
          if (e.fill.length) bake({ kind: "fill", h: hue(find(e.r)), fill: e.fill });
          break;
        case "same":
          stage = "same"; count.same = e.asked.length;
          let shown = 0;  // the first few questions go in front of the cast
          for (const [a, b] of e.asked) {
            const A = chainSpan[a], B = chainSpan[b];
            if (!A || !B) continue;
            const [p, q] = A.first[0] <= B.first[0] ? [A.last, B.first] : [B.last, A.first];
            arcs.push({ kind: "same", p, q, t0: now, r: a });
            if (opts.head && shown++ < 3) {
              const d = document.createElement("div"); d.className = "dv-card dv-pair dv-in";
              const row = document.createElement("div");
              row.append(img(best(A.ks) || p, "#ffb020"), img(best(B.ks) || q, "#ffb020"));
              const sp = document.createElement("span"); sp.textContent = t("viz.pair.same"); sp.style.color = "#ffb020";
              d.append(row, sp); facesEl.prepend(d);
            }
          }
          break;
      }
      caption();
    }
    const fmt = (v) => v.toLocaleString();
    function caption() {
      const nP = order.length;
      const T = {
        follow: [t("viz.follow.title"),
          t("viz.follow.sub", { follows: fmt(count.follows), pieces: fmt(nP), checks: fmt(count.checks), found: fmt(count.found), landed: fmt(count.landed) })],
        faces: [t("viz.faces.title"),
          t("viz.faces.sub", { faces: fmt(count.faces), frontal: fmt(count.frontal) })],
        pairs: [t("viz.pairs.title"),
          t("viz.pairs.sub", { pairs: fmt(count.pairs || 0), merges: count.merges, refused: count.refused, asks: count.asks })],
        people: [t("viz.people.title", { n: real.size }),
          [people && Object.keys(people).length - real.size ? t("viz.people.fragments", { n: Object.keys(people).length - real.size }) : "",
           dropped.size ? t("viz.people.dropped", { n: dropped.size }) : ""].filter(Boolean).join(" · ")],
        cover: [t("viz.cover.title"),
          t("viz.cover.sub", { covers: count.covers, filled: fmt(count.filled) })],
        same: [count.same ? t("viz.same.title", { n: count.same }) : t("viz.same.done"),
          count.same ? t("viz.same.sub") : t("viz.same.none")],
      }[stage] || [t("viz.start.title"), ""];
      titleEl.textContent = T[0]; subEl.textContent = T[1];
      legendEl.innerHTML = stage === "follow" ? `<i style="background:#3ddc84"></i>${t("viz.face.found")} <i style="background:#5b8cff"></i>${t("viz.legend.looks_same")} <i class="ring"></i>${t("viz.legend.nothing")}`
        : stage === "faces" ? `<i style="background:#fff"></i>${t("viz.face.clear")} <i style="background:#888"></i>${t("viz.legend.not_comparable")}`
        : stage === "pairs" ? `<i style="background:#3ddc84"></i>${t("viz.pair.merge")} <i style="background:#ff5c5c"></i>${t("viz.pair.refuse")} <i style="background:#ffb020"></i>${t("viz.pair.ask")}`
        : stage === "same" && count.same ? `<i style="background:#ffb020"></i>${t("viz.pair.same")}` : "";
    }

    // ---- drawing
    function color(k, now) {
      const r = find(+k);
      if (people && dropped.has(String(chainOf[k] ?? k))) return "rgba(120,125,135,0.18)";
      if (people && !real.has(String(r))) return `hsla(${hue(r)},25%,55%,0.4)`;  // a fragment: covered, nobody
      const solo = r === +k && !joined.has(String(k)) && !people;
      if (solo && stage === "follow") return "rgba(170,178,190,0.55)";
      const f = Math.min(1, (now - recolorAt) / 500);
      return `hsla(${hue(r)},${55 + 20 * f}%,${58 + 4 * f}%,0.9)`;
    }
    function draw() {
      if (stopped) return;
      raf = requestAnimationFrame(draw);
      const now = performance.now();
      // play queued events: on time, or faster when they've piled up (stay at most ~2.5 s behind)
      let backlog = queue.reduce((a, e) => a + (DWELL[e.e] ?? 100), 0);
      while (queue.length && now >= nextAt) {
        const e = queue.shift(); apply(e);
        const speed = Math.max(1, backlog / 2500);
        backlog -= DWELL[e.e] ?? 100;
        nextAt = Math.max(nextAt, now - 50) + (DWELL[e.e] ?? 100) / speed;
      }
      if (!layer.width || !layer.height) return;  // not laid out yet
      ctx.clearRect(0, 0, W, H);
      ctx.globalAlpha = stage === "follow" || !stage ? 1 : Math.max(0.12, 1 - (now - trailsFadeAt) / 1500);
      ctx.drawImage(tlayer, 0, 0, W, H); ctx.globalAlpha = 1;
      ctx.drawImage(layer, 0, 0, W, H);
      // grid: a light line every second of video
      ctx.strokeStyle = "rgba(255,255,255,0.05)"; ctx.lineWidth = 1;
      ctx.strokeRect(pad.l, pad.t, W - pad.l - pad.r, H - pad.t - pad.b);
      // pieces, sweeping in from the left when they arrive
      const sweep = Math.min(1, (now - piecesAt) / 900) * n;
      ctx.lineCap = "round"; ctx.lineJoin = "round";
      for (const k of order) {
        const obs = pieces[k];
        if (!obs.length || obs[0][0] > sweep) continue;
        ctx.strokeStyle = color(k, now); ctx.lineWidth = width(obs[0][3]);
        ctx.beginPath();
        let prev = null;
        for (const o of obs) {
          if (o[0] > sweep) break;
          if (prev && o[0] - prev[0] <= 2) ctx.lineTo(X(o[0]), Y(o[1])); else ctx.moveTo(X(o[0]), Y(o[1]));
          prev = o;
        }
        if (obs.length === 1) ctx.lineTo(X(obs[0][0]) + 0.1, Y(obs[0][1]));
        ctx.stroke();
      }
      // joins: the gap between two pieces whose follows met
      for (const l of links) {
        const f = Math.min(1, (now - l.t0) / 400);
        ctx.strokeStyle = color(l.r, now); ctx.lineWidth = 2; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(X(l.a[0]), Y(l.a[1]));
        ctx.lineTo(X(l.a[0] + (l.b[0] - l.a[0]) * f), Y(l.a[1] + (l.b[1] - l.a[1]) * f)); ctx.stroke(); ctx.setLineDash([]);
      }
      // face measurements
      if (Object.keys(faces).length) {
        for (const [key, lv] of Object.entries(faces)) {
          if (!lv) continue;
          const [k, t] = key.split(":"), o = at[k]?.[t];
          if (!o) continue;
          if (lv === 1 && stage !== "faces") continue;  // once measured, only the comparable faces stay, faintly
          ctx.fillStyle = lv === 2 ? (stage === "faces" ? "#fff" : "rgba(255,255,255,0.35)") : "rgba(200,200,200,0.45)";
          ctx.beginPath(); ctx.arc(X(+t), Y(o[1]), lv === 2 ? 2.3 : 1.4, 0, 7); ctx.fill();
        }
      }
      if (scanT != null) {
        const x = X(scanT), a = Math.max(0, 1 - (now - scanAt) / 1500);
        const g = ctx.createLinearGradient(x - 26, 0, x, 0);
        g.addColorStop(0, "rgba(255,255,255,0)"); g.addColorStop(1, `rgba(255,255,255,${0.16 * a})`);
        ctx.fillStyle = g; ctx.fillRect(x - 26, pad.t, 26, H - pad.t - pad.b);
        ctx.strokeStyle = `rgba(255,255,255,${0.6 * a})`; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke();
      }
      // follows in progress: one frame per 45 ms, the head's dot leading
      anims = anims.filter((a) => {
        const steps = a.p.x.length, shown = Math.min(steps, Math.floor((now - a.t0) / 45));
        for (let i = 0; i < shown; i++) {
          const c = HOW[a.p.how[i]] || "#888", t = a.p.from + a.dir * (i + 1);
          if (a.p.how[i] === "h" || a.p.how[i] === "x") {
            ctx.strokeStyle = c; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(X(t), Y(a.p.x[i]), 2.2, 0, 7); ctx.stroke();
          } else { ctx.fillStyle = c; ctx.beginPath(); ctx.arc(X(t), Y(a.p.x[i]), i === shown - 1 ? 3.4 : 2.2, 0, 7); ctx.fill(); }
        }
        if (shown >= steps) {
          const end = now - a.t0 - steps * 45;
          if (a.p.met != null && steps && end < 700) {  // landed on another piece: a ring at the landing
            const i = steps - 1, t = a.p.from + a.dir * steps;
            ctx.strokeStyle = `rgba(61,220,132,${1 - end / 700})`; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.arc(X(t), Y(a.p.x[i]), 5 + end / 40, 0, 7); ctx.stroke();
          }
          if (end >= 700 || !steps) { if (steps) bake({ kind: "trail", p: a.p, dir: a.dir }); return false; }
        }
        return true;
      });
      // comparisons: arcs from where one piece ends to where the other starts
      arcs = arcs.filter((a) => {
        // merges and refusals flash; the graph's unsure pairs stay until the people are settled; questions stay
        const life = a.kind === "merge" ? 1400 : a.kind === "refuse" ? 1600 : a.kind === "ask" && a.until ? a.until - a.t0 : Infinity;
        const age = now - a.t0;
        if (age > life) return false;
        const f = Math.min(1, age / 450), fade = life === Infinity ? 1 : Math.min(1, (life - age) / 400);
        const c = { merge: "61,220,132", refuse: "255,92,92", ask: "255,176,32", same: "255,176,32" }[a.kind];
        const x1 = X(a.p[0]), y1 = Y(a.p[1]), x2 = X(a.q[0]), y2 = Y(a.q[1]);
        const mx = (x1 + x2) / 2, my = Math.max(pad.t + 8, Math.min(y1, y2) - 18 - Math.abs(x2 - x1) * 0.15);
        ctx.strokeStyle = `rgba(${c},${0.95 * fade})`; ctx.lineWidth = 2;
        if (a.kind === "ask" || a.kind === "same") ctx.setLineDash([5, 4]);
        ctx.beginPath(); ctx.moveTo(x1, y1);
        const N = 24;
        for (let i = 1; i <= N * f; i++) {
          const s = i / N, x = (1 - s) * (1 - s) * x1 + 2 * s * (1 - s) * mx + s * s * x2, y = (1 - s) * (1 - s) * y1 + 2 * s * (1 - s) * my + s * s * y2;
          ctx.lineTo(x, y);
        }
        ctx.stroke(); ctx.setLineDash([]);
        if (f >= 1 && a.kind !== "merge") {
          const lx = 0.25 * x1 + 0.5 * mx + 0.25 * x2, ly = 0.25 * y1 + 0.5 * my + 0.25 * y2;
          ctx.fillStyle = `rgba(${c},${fade})`; ctx.font = "bold 13px -apple-system, sans-serif"; ctx.textAlign = "center";
          ctx.fillText(a.kind === "refuse" ? "×" : "?", lx, ly - 4);
        }
        return true;
      });
    }
    const ro = new ResizeObserver(resize); ro.observe(root);
    resize(); caption(); draw();
    return {
      push(evs) { queue.push(...evs); },
      // resolves once everything received has been shown (or after `max` ms)
      drained(max = 3000) {
        const t0 = performance.now();
        return new Promise((res) => { const tick = () => (!queue.length && !anims.length) || performance.now() - t0 > max ? res() : setTimeout(tick, 100); tick(); });
      },
      stop() { stopped = true; cancelAnimationFrame(raf); ro.disconnect(); },
    };
  }
  return { mount };
})();
