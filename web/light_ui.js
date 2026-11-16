// Scene-matched emoji lighting for the review tool: live preview + controls panel.
// The maths mirrors lighting.py (apply_settings / lit_sprite) so the preview matches the export;
// motion blur is previewed too (same velocity + shutter as the export); grain is export-only.
// Settings persist in edits.light (undoable like any edit).
// Uses review.html's globals: edits, sync, draw, afterDraw, resolved, zoom, cur, emojiSprite; icons.js's fillRange.
(() => {
  const DEFAULTS = { enabled: true, level: 1.0, mix: 1.0, amb: 0.15, ramp: 0.8, floor: 0.12,
                     light_colour: null, grain: true, blur: true, shutter: 1.0,
                     saturation: 1.0, black: 1.0 };
  let LD = null;          // /api/light payload
  const st = () => Object.assign({}, DEFAULTS, LD && LD.defaults, edits && edits.light);  // own defaults first: a newer page on an older server

  // ---- colour
  const TO_LIN = new Float32Array(256).map((_, i) => { const x = i / 255; return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4; });
  const TO_SRGB = new Uint8ClampedArray(4096).map((_, i) => { const x = i / 4095; return 255 * (x <= 0.0031308 ? x * 12.92 : 1.055 * x ** (1 / 2.4) - 0.055); });
  const srgb = (x) => TO_SRGB[x >= 1 ? 4095 : x <= 0 ? 0 : (x * 4095) | 0];
  const wheelSat = () => (LD && LD.wheel_sat) || 0.8;  // lighting.WHEEL_SAT
  // Light colour wheel [hue, radius] -> RGB normalised to mean 1: HSV(hue, radius x WHEEL_SAT, 1) (lighting.wheel_bgr)
  function wheelRGB([h, a]) {
    const sat = Math.min(Math.max(a, 0), 1) * wheelSat();
    const f = (n) => { const k = (n + h / 60) % 6; return 1 - sat * Math.max(0, Math.min(k, 4 - k, 1)); };
    const c = [f(5), f(3), f(1)], m = (c[0] + c[1] + c[2]) / 3;
    return c.map((v) => v / m);
  }
  // ...and back: where a light colour sits on the wheel (to show the estimated light as the wheel's start)
  function rgbWheel(c) {
    const mx = Math.max(...c), v = c.map((x) => x / mx), mn = Math.min(...v), d = 1 - mn;
    if (d < 1e-4) return [0, 0];
    const [r, g, b] = v;
    let h = r === 1 ? ((g - b) / d) % 6 : g === 1 ? (b - r) / d + 2 : (r - g) / d + 4;
    h = (h * 60 + 360) % 360;
    return [Math.round(h), Math.min(1, d / wheelSat())];
  }
  // The light colour the scene gives frame t (shared by every face; lighting.py cast_s), linear RGB with max 1
  function frameLight(t) {
    const k = LD && LD.scene_key[Math.min(Math.max(t | 0, 0), LD.scene_key.length - 1)];
    if (!k) return [1, 1, 1];
    const m = Math.max(...k) || 1;
    return k.map((v) => v / m);
  }
  function lightFor(piece, t, s) {
    const P = LD.pieces[piece];
    let p = P && P[t];
    // no estimate for this frame: the piece's nearest one within NEAR_FRAMES (lighting.py; 15 from a server that doesn't say)
    const near = (LD && LD.near_frames) || 15;
    for (let d = 1; P && !p && d <= near; d++) p = P[t - d] || P[t + d];
    const k0 = p ? p.slice(0, 3) : LD.scene_key[t], g = p ? p.slice(3, 5) : [0, 0];
    let key = k0.map((v) => s.level * v);
    if (s.light_colour) {  // a picked light colour: the estimate's brightness in that colour (lighting.apply_settings)
      const m = (key[0] + key[1] + key[2]) / 3, w = wheelRGB(s.light_colour);
      key = w.map((v) => m * v);
    }
    const amb = LD.scene_amb[t].map((v) => s.amb * v);
    return { key, amb, grad: g.map((v) => v * s.ramp) };
  }
  // the Saturation slider, and the footage's black level x the user's multiplier (lighting.film_settings);
  // a server without the film estimate gives no lift
  function filmFor(t, s) {
    const b0 = (LD.film && LD.film[t] && LD.film[t][1]) || 0;
    return { colour: Math.min(Math.max(+s.saturation, 0), 1), black: Math.min(Math.max(b0 * s.black, 0), 0.5) };
  }

  // ---- sprites: emoji glyph rasterised like review.html draws it, cached per (char, size)
  const sprites = new Map();
  const work = document.createElement("canvas"), wctx = work.getContext("2d", { willReadFrequently: true });
  function sprite(ch, N) {
    const k = ch + "|" + N;
    if (sprites.has(k)) return sprites.get(k);
    const c = document.createElement("canvas"); c.width = c.height = N;
    const g = c.getContext("2d", { willReadFrequently: true });
    const img = window.emojiSprite && window.emojiSprite(ch);  // the export's sprite (review.html), else text
    if (img) g.drawImage(img, 0, 0, N, N);
    else {
      g.font = `${N * 0.86}px "Apple Color Emoji","Segoe UI Emoji","Noto Color Emoji",sans-serif`;
      g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(ch, N / 2, N / 2 + N * 0.04);
    }
    const d = g.getImageData(0, 0, N, N).data, alb = new Float32Array(N * N * 3), a = new Uint8ClampedArray(N * N);
    for (let i = 0; i < N * N; i++) { alb[i * 3] = TO_LIN[d[i * 4]]; alb[i * 3 + 1] = TO_LIN[d[i * 4 + 1]]; alb[i * 3 + 2] = TO_LIN[d[i * 4 + 2]]; a[i] = d[i * 4 + 3]; }
    const sp = { N, alb, a };
    if (sprites.size > 400) sprites.clear();
    if (img) sprites.set(k, sp);  // a text stand-in isn't kept: the real sprite replaces it once loaded
    return sp;
  }

  // Returns false when there's nothing to do (lighting off or not ready, and full saturation) so review.html
  // falls back to its flat sprite. Saturation is about the emoji, so it applies even without lighting.
  function drawEmoji(ctx, ch, piece, t, x, y, size) {
    const s = st(), lit = !!(LD && s.enabled), sat = Math.min(Math.max(+s.saturation, 0), 1);
    if (!lit && sat >= 1) return false;
    const N = Math.max(16, Math.min(160, Math.round(size / 4) * 4));
    // without lighting: the flat emoji (mix 0), only desaturated
    const { key, amb, grad } = lit ? lightFor(piece, t, s) : { key: [1, 1, 1], amb: [0, 0, 0], grad: [0, 0] };
    const { colour, black } = lit ? filmFor(t, s) : { colour: sat, black: 0 };
    const sp = sprite(ch, N);
    if (work.width !== N) { work.width = work.height = N; }
    const img = wctx.createImageData(N, N), o = img.data, mix = lit ? s.mix : 0, fl = s.floor;
    for (let j = 0, i = 0; j < N; j++) {
      const yy = (j / (N - 1)) * 2 - 1;
      for (let q = 0; q < N; q++, i++) {
        const al = sp.a[i]; if (!al) continue;
        const r = Math.max(0.2, Math.min(2, 1 + grad[0] * ((q / (N - 1)) * 2 - 1) + grad[1] * yy));
        // film: towards the albedo's own luminance by the footage's colourfulness (lighting.desat)
        const a0 = sp.alb[i * 3], a1 = sp.alb[i * 3 + 1], a2 = sp.alb[i * 3 + 2], Y = 0.2126 * a0 + 0.7152 * a1 + 0.0722 * a2;
        for (let c = 0; c < 3; c++) {
          const A = Math.max(0, Y + colour * (sp.alb[i * 3 + c] - Y));
          const litC = Math.max(A * (key[c] * r + amb[c]), A * fl);
          o[i * 4 + c] = srgb(black + ((1 - mix) * A + mix * litC) * (1 - black));
        }
        o[i * 4 + 3] = al;
      }
    }
    wctx.putImageData(img, 0, 0);
    // motion blur: average copies along the motion vector (additive 'lighter' on premultiplied pixels
    // = a box filter, like lighting.motion_kernel); length = velocity x shutter, centred on the box
    const sh = lit ? shutterOf(s) : 0, [vx, vy] = sh > 0 ? velocity(piece, t) : [0, 0];  // no blur without lighting (as the export)
    const Lx = vx * sh * zoom, Ly = vy * sh * zoom, L = Math.hypot(Lx, Ly);
    if (L < 1) { ctx.drawImage(work, x - size / 2, y - size / 2, size, size); return true; }
    const K = Math.min(32, Math.max(3, Math.ceil(L / 1.5))), W = Math.ceil(size + L + 4);
    if (blurC.width !== W) { blurC.width = blurC.height = W; } else bctx.clearRect(0, 0, W, W);
    bctx.globalCompositeOperation = "lighter"; bctx.globalAlpha = 1 / K;
    for (let k = 0; k < K; k++) {
      const f = k / (K - 1) - 0.5;
      bctx.drawImage(work, W / 2 - size / 2 + f * Lx, W / 2 - size / 2 + f * Ly, size, size);
    }
    bctx.globalCompositeOperation = "source-over"; bctx.globalAlpha = 1;
    ctx.drawImage(blurC, x - W / 2, y - W / 2);
    return true;
  }
  const blurC = document.createElement("canvas"), bctx = blurC.getContext("2d");
  const shutterOf = (s) => (s.blur === false ? 0 : +s.shutter || 0);
  function velocity(piece, t) {  // from the resolved boxes on screen (matches FrameLighter.velocity)
    const at = (u) => { const b = (resolved[u] || []).find((it) => it[5] === piece); return b ? [b[1], b[2]] : null; };
    const c = at(t); if (!c) return [0, 0];
    const p = at(t - 1) || c, q = at(t + 1) || c;
    return [(q[0] - p[0]) / 2, (q[1] - p[1]) / 2];
  }

  // ---- panel
  const sec = document.createElement("section");
  sec.id = "lightsec";
  sec.innerHTML = `
    <h2>${t("light.title")} <span id="l_status" class="muted" style="text-transform:none;letter-spacing:0"></span></h2>
    <label data-tip="enabled" style="display:flex;justify-content:space-between;align-items:center;font-weight:600;color:var(--text)"><span>${t("light.enabled")}</span><input type="checkbox" class="switch" id="l_enabled"></label>
    <div style="display:flex;gap:8px;align-items:center;justify-content:space-between;margin:10px 0 6px">
      <span data-tip="light_colour" style="cursor:help;color:var(--soft)">${t("light.light_colour")}</span>
      <span style="display:flex;gap:8px;align-items:center">
        <span id="l_lc_seg" class="seg">
          <button id="l_lc_follow" data-tip="lc_follow">${t("light.follow")}</button>
          <button id="l_lc_fixed" data-tip="lc_fixed">${t("light.fixed")}</button>
        </span>
        <span id="l_lc_swatch" data-tip="lc_follow" style="width:16px;height:16px;border-radius:50%;border:1px solid #FFFFFF66;flex-shrink:0"></span>
      </span>
    </div>
    <canvas id="l_light_colour" data-tip="lc_fixed" width="96" height="96" style="display:none;margin:2px 0 8px auto;width:96px;height:96px;cursor:crosshair;touch-action:none"></canvas>
    <div id="l_sliders" style="display:grid;grid-template-columns:78px 1fr 38px;gap:10px 10px;align-items:center;color:var(--soft)"></div>
    <div style="display:flex;gap:10px;align-items:center;margin-top:12px;color:var(--soft)">
      <label data-tip="grain" style="display:flex;gap:9px;align-items:center"><input type="checkbox" id="l_grain"> <span>${t("light.grain")}</span></label><span style="flex:1"></span><button id="l_reset" class="small">${t("light.reset")}</button>
    </div>`;
  const anchor = document.getElementById("sel");
  anchor.parentNode.insertBefore(sec, anchor.nextSibling);
  const SLIDERS = [["mix", t("light.mix"), 0, 1, 0.05], ["level", t("light.level"), 0.3, 2.0, 0.05],
                   ["ramp", t("light.ramp"), 0, 1.5, 0.05], ["amb", t("light.amb"), 0, 0.6, 0.01],
                   ["saturation", t("light.saturation"), 0, 1, 0.05], ["black", t("light.black"), 0, 3, 0.05],
                   ["shutter", t("light.shutter"), 0, 2, 0.05]];
  const PCT = new Set(["mix", "saturation", "black"]);  // shown as %
  // Popover text per control: [title, what it does, what the ends mean].
  const TIPS = Object.fromEntries(["enabled", "light_colour", "lc_follow", "lc_fixed", "mix", "level", "ramp", "amb", "saturation", "black", "shutter", "grain"]
    .map((k) => [k, [t(`light.tip.${k}.title`), t(`light.tip.${k}.what`), t(`light.tip.${k}.ends`)]]));
  function tipText(k) {
    const [title, what, ends] = TIPS[k];
    return `<b>${title}</b><div style="margin-top:4px">${what}</div><div class="muted" style="margin-top:4px">${ends}</div>`;
  }
  const pop = Object.assign(document.createElement("div"), { id: "l_pop" });
  pop.style.cssText = "position:fixed;z-index:50;width:240px;padding:10px 12px;border-radius:10px;background:var(--menu);border:1px solid var(--line3);" +
    "box-shadow:0 12px 32px rgba(0,0,0,.6);font-size:12px;line-height:1.45;color:var(--text);pointer-events:none;display:none";
  document.body.append(pop);
  function showTip(el) {
    const k = el.dataset.tip; if (!TIPS[k]) return;
    pop.innerHTML = tipText(k); pop.style.display = "block";
    const r = el.getBoundingClientRect(), panel = sec.getBoundingClientRect(), w = pop.offsetWidth;
    // beside the panel (it sits at the window's right edge), level with the control; below it if there's no room
    if (panel.left - w - 10 >= 4) { pop.style.left = panel.left - w - 10 + "px"; pop.style.top = Math.min(r.top, innerHeight - pop.offsetHeight - 4) + "px"; }
    else { pop.style.left = Math.max(4, Math.min(r.left, innerWidth - w - 4)) + "px"; pop.style.top = r.bottom + 6 + "px"; }
  }
  const hideTip = () => { pop.style.display = "none"; };
  const sl = document.getElementById("l_sliders");
  for (const [k, name, lo, hi, step] of SLIDERS)
    sl.insertAdjacentHTML("beforeend", `<span data-tip="${k}" style="cursor:help">${name}</span><input type="range" id="l_${k}" data-tip="${k}" min="${lo}" max="${hi}" step="${step}"><span id="l_${k}_v" class="muted num" style="text-align:right"></span>`);
  for (const el of sec.querySelectorAll("[data-tip]")) {
    el.addEventListener("mouseenter", () => showTip(el)); el.addEventListener("mouseleave", hideTip);
    el.addEventListener("focus", () => showTip(el)); el.addEventListener("blur", hideTip);
  }

  // Edits: live-update while dragging, save on release (one step on the server's undo history).
  let before = null;
  function begin() { if (before === null) before = JSON.stringify(edits); }
  function set(k, v) { begin(); edits.light = Object.assign({}, edits.light, { [k]: v }); draw(); }
  function end() {
    if (before === null) return;
    if (before !== JSON.stringify(edits)) sync();
    before = null;
  }

  for (const [k] of SLIDERS) {
    const el = document.getElementById("l_" + k);
    el.addEventListener("input", () => { set(k, +el.value); if (k === "shutter") edits.light.blur = true; });
    el.addEventListener("change", end);
  }
  for (const k of ["enabled", "grain"]) {
    const el = document.getElementById("l_" + k);
    el.addEventListener("change", () => { set(k, el.checked); end(); });
  }
  document.getElementById("l_reset").onclick = () => { begin(); delete edits.light; draw(); end(); };

  function wheel(id) {
    const cv = document.getElementById(id), g = cv.getContext("2d"), R = 44, C = 48;
    function paint(h, a) {
      g.clearRect(0, 0, 96, 96);
      const cg = g.createConicGradient(0, C, C);
      for (let i = 0; i <= 6; i++) cg.addColorStop(i / 6, `hsl(${i * 60},100%,50%)`);
      g.fillStyle = cg; g.beginPath(); g.arc(C, C, R, 0, 2 * Math.PI); g.fill();
      const rg = g.createRadialGradient(C, C, 0, C, C, R);
      rg.addColorStop(0, "rgba(235,235,235,1)"); rg.addColorStop(1, "rgba(235,235,235,0)");  // centre = white light
      g.fillStyle = rg; g.beginPath(); g.arc(C, C, R, 0, 2 * Math.PI); g.fill();
      const x = C + Math.cos(h * Math.PI / 180) * a * R, y = C + Math.sin(h * Math.PI / 180) * a * R;
      g.lineWidth = 2; g.strokeStyle = "#000"; g.beginPath(); g.arc(x, y, 5, 0, 2 * Math.PI); g.stroke();
      g.strokeStyle = "#fff"; g.lineWidth = 1.5; g.beginPath(); g.arc(x, y, 5, 0, 2 * Math.PI); g.stroke();
    }
    const pick = (ev) => {
      const r = cv.getBoundingClientRect(), dx = ev.clientX - r.left - C, dy = ev.clientY - r.top - C;
      const a = Math.min(Math.hypot(dx, dy) / R, 1), h = (Math.atan2(dy, dx) * 180 / Math.PI + 360) % 360;
      set(id.slice(2), [Math.round(h), Math.round(a * 100) / 100]);
    };
    cv.addEventListener("pointerdown", (ev) => { cv.setPointerCapture(ev.pointerId); pick(ev); });
    cv.addEventListener("pointermove", (ev) => { if (cv.hasPointerCapture(ev.pointerId)) pick(ev); });
    cv.addEventListener("pointerup", end);
    cv.addEventListener("dblclick", () => { set(id.slice(2), [0, 0]); end(); });  // white light
    return paint;
  }
  const paintLight = wheel("l_light_colour");
  document.getElementById("l_lc_follow").onclick = () => { set("light_colour", null); end(); };
  // Fixed starts at this frame's scene colour, so switching doesn't change the picture; then drag the wheel
  document.getElementById("l_lc_fixed").onclick = () => { if (!st().light_colour) { set("light_colour", rgbWheel(frameLight(cur))); end(); } };
  const swatch = document.getElementById("l_lc_swatch");
  let swatchT = -1;
  function syncSwatch(force) {  // this frame's scene light, while following the scene (changes as you scrub/play)
    if (!LD || (!force && cur === swatchT)) return;
    swatchT = cur;
    const c = frameLight(cur).map(srgb);
    swatch.style.background = `rgb(${c.join(",")})`;
  }

  // Reflect current settings (after undo, load, reset) in the controls.
  let shown = "";
  function syncPanel() {
    if (!edits) return;
    const s = st(), j = JSON.stringify([s, !!LD]);
    syncSwatch(false);
    if (j === shown) return;
    shown = j;
    const val = (k) => (k === "shutter" ? shutterOf(s) : s[k]);
    for (const [k] of SLIDERS) { document.getElementById("l_" + k).value = val(k); fillRange(document.getElementById("l_" + k)); document.getElementById(`l_${k}_v`).textContent = PCT.has(k) ? Math.round(val(k) * 100) + "%" : (+val(k)).toFixed(2); }
    for (const k of ["enabled", "grain"]) document.getElementById("l_" + k).checked = s[k];
    const fixed = !!s.light_colour;
    document.getElementById("l_lc_follow").classList.toggle("on", !fixed);
    document.getElementById("l_lc_fixed").classList.toggle("on", fixed);
    document.getElementById("l_light_colour").style.display = fixed ? "block" : "none";
    swatch.style.visibility = fixed ? "hidden" : "visible";
    if (fixed) paintLight(...s.light_colour);
    syncSwatch(true);
    sec.style.opacity = s.enabled ? 1 : 0.75;
  }
  afterDraw(syncPanel);

  async function load() {
    const status = document.getElementById("l_status");
    const disable = (msg) => {
      status.textContent = ""; sec.style.opacity = 0.45; sec.style.pointerEvents = "none";
      sec.insertAdjacentHTML("beforeend", `<div style="color:var(--coast);font-size:12px;margin-top:6px;pointer-events:auto">${msg}</div>`);
    };
    try {
      const r = await (await fetch("api/light")).json();
      if (r.ready) { LD = r; status.textContent = ""; sprites.clear(); draw(); return; }
      if (r.error) return disable(t("light.err.failed", { error: r.error }));
      status.textContent = t("light.estimating", { pct: Math.round(r.progress * 100) });
      setTimeout(load, 2000);
    } catch (e) { disable(t("light.err.unavailable", { error: e.message })); }
  }
  load();
  window.LightUI = { drawEmoji, dragging: () => before !== null };
})();
