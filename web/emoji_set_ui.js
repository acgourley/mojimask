// "Presets" panel for the review tool (mojimask cover style): pick which emoji people get. Presets come
// from palettes.json (page_util.js loads it; shared with palettes.py, which applies the same choice in the export)
// and can't be edited; "Custom" is the one you can change: it starts as just 🥸, and you add or drop emoji there.
// It's saved with the project and kept when you switch to a preset and back. Stored in edits.palette (+ edits.palette_custom, kept when you switch away), undoable.
// Uses review.html's globals: edits, commit, draw, afterDraw, api, refreshEdits, renderSel; and page_util.js's
// PRESETS, loadPresets, presetByKey, presetKey, isCustom, paletteFor.
(() => {
  const seg = new Intl.Segmenter("en", { granularity: "grapheme" });
  const parseEmoji = (s) => [...seg.segment(s)].map((x) => x.segment).filter((g) => /\p{Extended_Pictographic}|\p{Regional_Indicator}/u.test(g));

  const sec = document.createElement("section");
  sec.id = "emojiset";
  sec.innerHTML = `
    <h2>${t("emojiset.title")}</h2>
    <select id="es_pick" aria-label="${t("emojiset.label")}" style="width:100%"></select>
    <div id="es_hint" class="muted" style="font-size:12px;margin-top:6px"></div>
    <div id="es_chips" style="display:flex;flex-wrap:wrap;gap:6px 8px;margin:10px 0 10px;font-size:23px;line-height:1"></div>
    <div id="es_edit">
      <input id="es_add" placeholder="${t("emojiset.add_placeholder")}" style="width:100%">
      <div class="muted" style="font-size:11px;margin:4px 0 8px">${t("emojiset.drop_hint")}</div>
    </div>
    <div id="es_over" style="font-size:12px;margin-bottom:8px"></div>
    <label style="display:flex;gap:9px;align-items:center;font-size:13px;color:var(--soft)" title="${t("emojiset.pose_tip")}">
      <input type="checkbox" id="es_pose"> <span>${t("emojiset.pose")}</span></label>`;
  const anchor = document.getElementById("sel");
  anchor.parentNode.insertBefore(sec, anchor.nextSibling);
  const posebox = sec.querySelector("#es_pose");
  posebox.addEventListener("change", () => commit((e) => { if (posebox.checked) e.head_pose = true; else delete e.head_pose; }));
  const CUSTOM_START = ["🥸"];
  const pick = sec.querySelector("#es_pick"), chips = sec.querySelector("#es_chips"), add = sec.querySelector("#es_add"),
        over = sec.querySelector("#es_over"), hint = sec.querySelector("#es_hint"), editBox = sec.querySelector("#es_edit");

  const setCustom = (list) => commit((e) => { e.palette = "custom"; e.palette_custom = list; });
  pick.addEventListener("change", () => {
    const k = pick.value;
    // Custom starts as just the disguise face (or the custom mix you made before).
    commit((e) => { e.palette = k; if (k === "custom" && !(e.palette_custom && e.palette_custom.length)) e.palette_custom = [...CUSTOM_START]; });
  });
  add.addEventListener("keydown", (ev) => {
    if (ev.key !== "Enter" || !isCustom(edits)) return;
    const extra = parseEmoji(add.value), cur = paletteFor(edits);
    add.value = "";
    if (extra.length) setCustom([...cur, ...extra.filter((x) => !cur.includes(x))]);
  });
  window.EmojiSetDrop = (i) => { if (!isCustom(edits)) return; const cur = [...paletteFor(edits)]; if (cur.length <= 1) return; cur.splice(i, 1); setCustom(cur); };
  // back to the preset's own deal for everyone: each pick made under this preset is taken back (as facts: the server
  // owns edits.emoji, worked out from them)
  window.EmojiSetClearOverrides = async () => {
    const facts = Object.keys(edits.emoji || {}).map((tid) => ({ kind: "emoji", value: "", pieces: (edits.people[String(tid).replace(/^@/, "")] || {}).pieces || [] })).filter((f) => f.pieces.length);
    if (!facts.length) return;
    await api("api/facts", { facts, label: "Clear picked emoji", page: "edit" });
    await refreshEdits();
  };

  let shown = "";
  function syncPanel() {
    if (!PRESETS || !edits) return;
    const cur = paletteFor(edits), custom = isCustom(edits), nOver = Object.keys(edits.emoji || {}).length;
    posebox.checked = !!edits.head_pose;
    const j = JSON.stringify([edits.palette, cur, nOver]);
    if (j === shown) return;
    shown = j;
    pick.innerHTML = PRESETS.map((p) => `<option value="${p.key}">${p.label}</option>`).join("") + `<option value="custom">${t("emojiset.custom")}</option>`;
    pick.value = custom ? "custom" : presetKey(edits);
    hint.textContent = custom ? t("emojiset.custom_hint") : presetByKey(presetKey(edits)).hint || "";
    editBox.style.display = custom ? "" : "none";
    chips.innerHTML = cur.map((e, i) => custom
      ? `<span title="${t("emojiset.drop_tip")}" onclick="EmojiSetDrop(${i})" style="cursor:pointer;padding:1px">${e}</span>`
      : `<span style="padding:1px">${e}</span>`).join("");
    over.innerHTML = nOver ? `<span class="muted">${t("emojiset.overrides", { n: nOver })}</span> <a href="#" onclick="EmojiSetClearOverrides();return false" style="color:var(--accent)">${t("emojiset.clear")}</a>` : "";
  }
  afterDraw(syncPanel);

  loadPresets().then((p) => { if (p) { draw(); renderSel(); } });
})();
