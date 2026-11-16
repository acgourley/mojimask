// Depth view for the editor (MOJIMASK_DEPTH on; button "Depth" / key D): the frame's depth map washed over the picture
// (blue far, red near), and for each cover what partial occlusion (depth.py, Occlusion.plan) decided and from what:
//   the square the emoji covers, and the face zone (dashed circle) where the strict margin applies;
//   a label: the face's depth around this frame, the fraction of the square reading in front of it now, and the
//   outcome -- "cut" (the video shows where something is in front), "not cut", or "self" (most of the face zone
//   reads in front: that's the face itself nearer than its estimate, so nothing is cut).
// Uses review.html globals: S, cur, ctx, zoom, draw, afterDraw; icons.js's Icon.
(() => {
  let on = false;
  const info = new Map(), pics = new Map();
  const btn = document.createElement("button");
  const face = () => { btn.innerHTML = `${Icon("layers", 14)}${t("edit.depth")} <span class="state">${on ? t("edit.toggle_on") : t("edit.toggle_off")}</span>`; btn.classList.toggle("on", on); btn.setAttribute("aria-pressed", on); };
  face(); btn.title = t("edit.depth_tip");
  document.getElementById("viewbtn").after(btn);
  const toggle = () => { on = !on; face(); draw(); };
  btn.onclick = toggle;
  window.addEventListener("keydown", (ev) => {
    if (ev.metaKey || ev.ctrlKey || ev.altKey || ["INPUT", "TEXTAREA", "SELECT"].includes(ev.target.tagName) || ev.target.isContentEditable) return;
    if (ev.key === "d" || ev.key === "D") toggle();
  });
  function get(t) {
    if (!info.has(t)) {
      if (info.size > 200) for (const k of info.keys()) { if (Math.abs(k - t) > 60) { info.delete(k); pics.delete(k); } if (info.size < 150) break; }
      info.set(t, null);
      const forget = () => { info.delete(t); pics.delete(t); };  // a failed load is tried again when the frame is next drawn
      fetch(`api/depth/${t}`).then((r) => r.json()).then((d) => { info.set(t, d); if (on && t === cur) draw(); }).catch(forget);
      const im = new Image(); im.onload = () => { if (on && t === cur) draw(); }; im.onerror = forget; im.src = `api/depth/${t}.jpg`;
      pics.set(t, im);
    }
    return [info.get(t), pics.get(t)];
  }
  function paint() {
    const [d, im] = get(cur);
    for (let u = cur + 1; u < Math.min(S.n, cur + 8); u++) get(u);  // read ahead for playback
    ctx.save();
    if (im && im.complete && im.naturalWidth) {
      ctx.globalAlpha = 0.55; ctx.drawImage(im, 0, 0, S.w * zoom, S.h * zoom); ctx.globalAlpha = 1;
    }
    if (d && d.covers) {
      ctx.font = "11px -apple-system, sans-serif"; ctx.lineWidth = 1.5;
      for (const c of d.covers) {
        const [x, y, s] = c.box, h = s * S.scale / 2;
        const what = c.face === undefined ? "drawn, never cut" : c.self ? "self" : c.cut ? "cut" : "not cut";
        const col = what === "cut" ? "#7CFC00" : what === "self" ? "#ff5aa0" : "#ffffff";
        ctx.strokeStyle = col;
        ctx.strokeRect((x - h) * zoom, (y - h) * zoom, 2 * h * zoom, 2 * h * zoom);
        // the face zone: depth.py's FACE_ZONE[0] head sizes around a point zone_dy (track.FACE_DY) head sizes below the box's centre
        const zr = d.face_zone ?? 0.4, zdy = d.zone_dy ?? 0.08;
        ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.arc(x * zoom, (y + zdy * s) * zoom, zr * s * zoom, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
        const txt = c.face === undefined ? what
          : `${what} · face ${c.face >= 0 ? "+" : ""}${c.face.toFixed(2)} · in front ${Math.round(c.in_front * 100)}% · margin ${c.margin}/${c.rim}`;
        const lx = (x - h) * zoom, ly = (y - h) * zoom - 4;
        ctx.fillStyle = "rgba(0,0,0,.65)"; ctx.fillRect(lx - 2, ly - 11, ctx.measureText(txt).width + 4, 14);
        ctx.fillStyle = col; ctx.fillText(txt, lx, ly);
      }
    }
    ctx.restore();
  }
  afterDraw(() => { if (on) paint(); });
})();
