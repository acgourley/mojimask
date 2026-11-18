// Appended to every page's strings.js by the screen catalog (run.py intercepts the file as the pages load it; the app
// itself is unchanged): which key each piece of text on screen came from, and, with --drafts, other wording laid over
// today's. __DRAFTS__ is replaced with the drafts file's contents ({} without --drafts):
//
//   {"strings": {"decisions.a.different_people": "No, or uncertain", ...},     strings.js keys -> draft text
//    "rules":   {"rule: you said these two are different people": "...", ...}, the rule texts the Python side sends
//               (shown in Decisions' tooltips): today's text, or the start of it, -> draft (the rest is kept, so a
//               rule with numbers after it still reads whole). Only where t() fills a {rule}.
//    ...}       anything else (notes) is left alone
const CATALOG_DRAFTS = __DRAFTS__;
Object.assign(STRINGS.en, CATALOG_DRAFTS.strings || {});
window.__tUsed = new Map();    // text -> key
window.__tFilled = new Set();  // texts with values filled in ({n}, a name): the viewer doesn't copy edits into them
// The key a piece of text on screen came from (measure.js, snapshot.js): the text itself, two strings joined by " · "
// ("5.3 s on screen · 2 joined" -> "a + b"), or the longest string it starts with or holds (a caption with more after it)
window.__tKeyOf = (text) => {
  const used = window.__tUsed;
  if (used.has(text)) return used.get(text);
  const parts = text.split(/\s+·\s+/);
  if (parts.length > 1) { const ks = [...new Set(parts.map((p) => used.get(p)).filter(Boolean))]; if (ks.length) return ks.join(" + "); }
  let best = null;
  for (const [s, k] of used) if (s.length >= 6 && (text.startsWith(s) || (s.length >= 12 && text.includes(s))) && (!best || s.length > best[0].length)) best = [s, k];
  return best ? best[1] : null;
};
t = ((orig) => function (key, vars) {
  if (vars && typeof vars.rule === "string") {
    for (const [now, draft] of Object.entries(CATALOG_DRAFTS.rules || {})) {
      if (vars.rule.startsWith(now)) { vars = { ...vars, rule: draft + vars.rule.slice(now.length) }; break; }
    }
  }
  const s = orig(key, vars);
  if (typeof s === "string" && s.trim()) { window.__tUsed.set(s.trim(), key); if (vars) window.__tFilled.add(s.trim()); }
  return s;
})(t);
