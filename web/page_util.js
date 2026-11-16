// Helpers the project pages share (review.html, people.html, decisions.html and emoji_set_ui.js), so each is
// written once. A plain script, loaded before the page's own: everything here is a global.

// ---- emoji sets: palettes.json, shared with palettes.py, which makes the same choice in the export.
// edits.palette is a preset key or "custom" (edits.palette_custom); an unknown key (a retired preset) and a
// missing one fall back to the default preset (the one palettes.json marks "default": true).
// Until palettes.json loads, everyone's set is the default preset's: the one fallback list.
const FALLBACK_EMOJI = ["😎", "🤩", "🥳", "😜", "🤪", "😝", "🤠", "🥸", "😏", "😆", "🤑", "😇", "🐶", "🐱", "🦊", "🐼",
  "🐨", "🐯", "🦁", "🐸", "🐵", "🐷", "🐮", "🐰", "🐻", "🐹", "🌝"];  // palettes.json's masquerave
let PRESETS = null;  // palettes.json's list, once loaded
let presetsLoading = null;
// palettes.json, fetched once per page: resolves to the list, or null if it couldn't be read
function loadPresets() {
  return (presetsLoading ||= fetch("palettes.json").then((r) => r.json()).then((p) => (PRESETS = p)).catch(() => null));
}
const presetByKey = (k) => (PRESETS && PRESETS.find((p) => p.key === k)) || null;
function presetKey(e) {  // the preset in use (custom aside): edits.palette if it's a preset, else the default
  if (presetByKey(e && e.palette)) return e.palette;
  return PRESETS ? (PRESETS.find((p) => p.default) || PRESETS[0]).key : null;
}
const isCustom = (e) => !!(e && e.palette === "custom" && e.palette_custom && e.palette_custom.length);
// the emoji people get under these edits (palettes.py for_edits)
function paletteFor(e) {
  if (isCustom(e)) return e.palette_custom;
  return PRESETS ? presetByKey(presetKey(e)).emoji : FALLBACK_EMOJI;
}
// the emoji a person or box wears: a hand pick (edits.emoji), else its slot in the set by a hash of its id (palettes.py
// pick: same person, same slot in whatever set is active). A person "@x" defaults to its anchor box x's emoji.
function emojiFor(tid, e) {
  if (e && e.emoji && e.emoji[tid]) return e.emoji[tid];
  let h = 0; const s = String(tid).replace(/^@/, "");
  for (let i = 0; i < s.length; i++) h += (i + 1) * s.charCodeAt(i);
  const P = paletteFor(e); return P[h % P.length];
}

// ---- people: edits.people = {pid: {name, pieces, show}}, resolved by the server to the stable id "@<pid>"
const pidOf = (tid) => (String(tid).startsWith("@") ? String(tid).slice(1) : null);
const personOf = (tid, e) => { const pid = pidOf(tid); return pid ? ((e && e.people) || {})[pid] : null; };

// ---- pictures
// frame t's picture (api/frames), loaded once: resolves to the image, or null if it didn't load
const FRAMES = new Map();
function frame(t) {
  if (!FRAMES.has(t)) FRAMES.set(t, new Promise((ok) => { const im = new Image(); im.onload = () => ok(im); im.onerror = () => ok(null); im.src = `api/frames/${String(t).padStart(5, "0")}.jpg`; }));
  return FRAMES.get(t);
}
// a canvas drawn at the screen's own pixels (2 per point on Retina): n points across and h down (square without h),
// drawn in points. (A new canvas is 300 x 150.)
function hidpi(cv, n, h = n) {
  const d = window.devicePixelRatio || 1, px = Math.round(n * d), py = Math.round(h * d);
  if (cv.width !== px || cv.height !== py) { cv.width = px; cv.height = py; }
  cv.dataset.n = n;
  const g = cv.getContext("2d"); g.setTransform(d, 0, 0, d, 0, 0); g.imageSmoothingQuality = "high";
  return g;
}
