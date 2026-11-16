// Pose debug overlay for the editor (button "Pose" / key B): draws every body skeleton the scan found on the
// current frame (keypoints and limbs), labelled with which way it faces; the dashed circle is the head the
// keypoints place (heads are placed from them).
// Uses review.html globals: S, cur, ctx, zoom, draw, afterDraw; icons.js's Icon.
(() => {
  let on = false;
  const cache = new Map();
  const EDGES = [[5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [0, 1], [0, 2], [1, 3], [2, 4]];
  const COLOR = "#bbbbbb", KP_MIN = 0.3;  // keypoints less sure than KP_MIN aren't drawn
  const btn = document.createElement("button");
  const face = () => { btn.innerHTML = `${Icon("person-standing", 14)}${t("edit.pose")} <span class="state">${on ? t("edit.toggle_on") : t("edit.toggle_off")}</span>`; btn.classList.toggle("on", on); btn.setAttribute("aria-pressed", on); };
  face(); btn.title = t("edit.pose_tip");
  document.getElementById("viewbtn").after(btn);
  const toggle = () => { on = !on; face(); draw(); };
  btn.onclick = toggle;
  window.addEventListener("keydown", (ev) => {
    if (ev.metaKey || ev.ctrlKey || ev.altKey || ["INPUT", "TEXTAREA", "SELECT"].includes(ev.target.tagName) || ev.target.isContentEditable) return;
    if (ev.key === "b" || ev.key === "B") toggle();
  });
  function get(t) {
    if (!cache.has(t)) {
      if (cache.size > 400) for (const k of cache.keys()) { if (Math.abs(k - t) > 100) cache.delete(k); if (cache.size < 300) break; }
      cache.set(t, null);
      fetch(`api/pose/${t}`).then((r) => r.json()).then((d) => { cache.set(t, d); if (on && t === cur) draw(); })
        .catch(() => cache.delete(t));  // tried again when the frame is next drawn
    }
    return cache.get(t);
  }
  function paint() {
    const ps = get(cur);
    if (on) for (let u = cur + 1; u < Math.min(S.n, cur + 15); u++) get(u);  // prefetch for playback
    if (!ps) return;
    ctx.save();
    for (const p of ps) {
      const k = p.kps, c = COLOR;
      ctx.strokeStyle = c; ctx.fillStyle = c; ctx.lineWidth = 2;
      for (const [i, j] of EDGES) {
        if (k[i * 3 + 2] > KP_MIN && k[j * 3 + 2] > KP_MIN) {
          ctx.beginPath(); ctx.moveTo(k[i * 3] * zoom, k[i * 3 + 1] * zoom); ctx.lineTo(k[j * 3] * zoom, k[j * 3 + 1] * zoom); ctx.stroke();
        }
      }
      for (let i = 0; i < 17; i++) if (k[i * 3 + 2] > KP_MIN) { ctx.beginPath(); ctx.arc(k[i * 3] * zoom, k[i * 3 + 1] * zoom, 2.5, 0, 2 * Math.PI); ctx.fill(); }
      if (p.head) {
        ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.arc(p.head[0] * zoom, p.head[1] * zoom, p.head[2] * zoom * 0.6, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
      }
      const lx = k[0 * 3] || k[5 * 3], ly = (k[0 * 3 + 1] || k[5 * 3 + 1]) - 10;
      ctx.font = "11px -apple-system, sans-serif";
      const txt = String(p.facing);
      ctx.fillStyle = "rgba(0,0,0,.6)"; ctx.fillRect(lx * zoom - 2, ly * zoom - 11, ctx.measureText(txt).width + 4, 14);
      ctx.fillStyle = c; ctx.fillText(txt, lx * zoom, ly * zoom);
    }
    ctx.restore();
  }
  afterDraw(() => { if (on) paint(); });
})();
